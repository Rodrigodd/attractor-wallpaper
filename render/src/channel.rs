//! A zero copy channel for sending messages between threads.

use std::{
    cell::UnsafeCell,
    panic::AssertUnwindSafe,
    ptr,
    sync::{
        atomic::{
            AtomicU8,
            Ordering::{Acquire, Relaxed, Release},
        },
        Arc,
    },
    thread::{self, Thread},
};

#[repr(u8)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum ChannelState {
    /// The channel is ready to send and receive messages.
    Ready,
    /// The receiver is waiting for the sender to send a message.
    RecvWaiting,
    /// The sender is waiting for the receiver to release the message.
    SenderWaiting,
    /// Eihter the sender or the receiver was dropped.
    Closed,
}

impl From<u8> for ChannelState {
    fn from(value: u8) -> Self {
        match value {
            0 => Self::Ready,
            1 => Self::RecvWaiting,
            2 => Self::SenderWaiting,
            3 => Self::Closed,
            _ => unreachable!(),
        }
    }
}

struct Channel<T> {
    state: AtomicU8,
    waiting_thread: AssertUnwindSafe<UnsafeCell<Thread>>,
    message: AssertUnwindSafe<UnsafeCell<*mut T>>,
}
unsafe impl<T: Sync + Send> Send for Channel<T> {}
unsafe impl<T: Sync + Send> Sync for Channel<T> {}

pub struct Sender<T: Sync + Send> {
    channel: Arc<Channel<T>>,
}
pub struct Receiver<T: Sync + Send> {
    channel: Arc<Channel<T>>,
}

impl<T: Sync + Send> Sender<T> {
    pub fn is_closed(&self) -> bool {
        let state: ChannelState = self.channel.state.load(Acquire).into();

        state == ChannelState::Closed
    }

    /// Send a reference to `value` to the receiver.
    ///
    /// If the receiver is not waiting for a message, this will do nothing. Otherwise it will
    /// notify the receiver that a message is ready, and block until the receiver releases it.
    ///
    /// if the receiver has been dropped, this will do nothing.
    pub fn send(&mut self, value: &mut T) {
        // 1. check if receiver is waiting

        let state: ChannelState = self.channel.state.load(Acquire).into();

        if state != ChannelState::RecvWaiting {
            // Invariant: The initial value of state is Ready. The sender only sets the state to
            // SenderWaiting, Ready or Closed. This function only returns with state=Ready or
            // state = Closed;
            debug_assert!(matches!(state, ChannelState::Ready | ChannelState::Closed));

            // if not, do nothing
            return;
        }

        // 2. if yes, put the message in the channel,

        let value = &mut *value;

        // SAFETY: this code is only reachable after observing RecvWaiting, which means that
        // the receiver is blocked, and cannot be borrowing the channel.
        unsafe {
            *self.channel.message.get() = value;
        };

        // 3. unblock the receiver,

        let recv_thread;
        {
            // SAFETY: this code is only reachable after observing RecvWaiting, which means that
            // the receiver has already filled waiting_thread with its thread.
            let waiting_thread = unsafe { &mut *self.channel.waiting_thread.get() };

            recv_thread = waiting_thread.clone();

            // make sure to unblock recv only after marking this as the waiting thread.
            *waiting_thread = thread::current();
        }

        self.channel
            .state
            .store(ChannelState::SenderWaiting as u8, Release);

        recv_thread.unpark();

        // 4. and wait until receiver releases the message
        loop {
            let state: ChannelState = self.channel.state.load(Acquire).into();

            // After releasing the message, there receiverc could already have started waiting for
            // the next one, or could have been dropped. So the state here can be Read, Closed or
            // RecvWaiting, but not SenderWaiting.
            if state != ChannelState::SenderWaiting {
                break;
            }
            thread::park();
        }
    }
}

impl<T: Sync + Send> Receiver<T> {
    /// Receive a message from the sender.
    ///
    /// This will block until the sender sends a message. Unless the sender is droped, in which
    /// case this will return `None`.
    pub fn recv<V>(&mut self, then: impl FnOnce(&mut T) -> V) -> Option<V> {
        let state: ChannelState = self.channel.state.load(Acquire).into();

        // Invariant: The initial state is Ready. The Sender only sets the state when Dropped or
        // when the Receiver is waiting in this function. This function only returns with
        // state=Ready or state=Closed.
        debug_assert!(matches!(state, ChannelState::Ready | ChannelState::Closed));

        if state == ChannelState::Closed {
            return None;
        }

        // 1. wait until sender sends the message

        {
            // SAFETY: the receiver only get a mutable reference to the channel after observing
            // RecvWaiting, and drops it before writing state to SenderWaiting.
            let waiting_thread = unsafe { &mut *self.channel.waiting_thread.get() };

            *waiting_thread = thread::current();
        }

        // the sender could have being dropped in the mean time, so check again in a compare
        // exchange.
        let result: Result<ChannelState, ChannelState> = self
            .channel
            .state
            .compare_exchange(
                ChannelState::Ready as u8,
                ChannelState::RecvWaiting as u8,
                Release,
                Acquire,
            )
            .map(|x| x.into())
            .map_err(|x| x.into());

        match result {
            Err(ChannelState::Closed) => {
                return None;
            }
            Err(_) => unsafe {
                // SAFETY: this code is only reachable after observing Ready and before writing
                // RecvWaiting. This means the Sender could only have set the state to Closed or
                // left as Ready. But it cannot be Ready, because the compare_exchange only fails
                // if it was written.
                std::hint::unreachable_unchecked()
            },
            Ok(_) => {}
        }

        loop {
            let state: ChannelState = self.channel.state.load(Acquire).into();
            if state == ChannelState::Closed {
                return None;
            }
            if state == ChannelState::SenderWaiting {
                break;
            }
            thread::park();
        }

        // 2. when message is received, call `then` with the message

        let unblock_sender = finally(|| {
            // 3. notify sender that message has been released

            let recv_thread;
            {
                let waiting_thread = unsafe { &mut *self.channel.waiting_thread.get() };

                recv_thread = waiting_thread.clone();
            }

            self.channel.state.store(ChannelState::Ready as u8, Release);

            recv_thread.unpark();
        });

        // SAFETY: this code is only reachable after observing SenderWaiting, which means that
        // the sender has already written to the message cell and dropped its reference.
        let t = unsafe { &mut **self.channel.message.get() };

        // The finally block above will ensure that the sender is still notified even if `then`
        // panics.
        let result = (then)(t);

        drop(unblock_sender);

        Some(result)
    }
}

impl<T: Sync + Send> Drop for Sender<T> {
    fn drop(&mut self) {
        loop {
            let state: ChannelState = self.channel.state.load(Acquire).into();

            if state == ChannelState::Closed {
                return;
            }

            if state == ChannelState::Ready {
                // mark channel as closed

                // Make sure that the channel doesn't transitioned to RecvWaiting in the meantime.
                let result = self.channel.state.compare_exchange_weak(
                    ChannelState::Ready as u8,
                    ChannelState::Closed as u8,
                    Relaxed,
                    Relaxed,
                );

                match result {
                    Ok(_) => return,
                    Err(_) => continue,
                }
            }

            debug_assert!(state == ChannelState::RecvWaiting);

            let recv_thread;
            {
                // SAFETY: this is only reachable after observing RecvWaiting, which means that the
                // receiver has already written to the waiting_thread cell.
                let waiting_thread = unsafe { &mut *self.channel.waiting_thread.get() };

                recv_thread = waiting_thread.clone();
            }

            // mark channel as closed
            self.channel
                .state
                .store(ChannelState::Closed as u8, Release);

            recv_thread.unpark();

            return;
        }
    }
}

impl<T: Sync + Send> Drop for Receiver<T> {
    fn drop(&mut self) {
        let state: ChannelState = self.channel.state.load(Acquire).into();

        match state {
            ChannelState::Closed => {}
            ChannelState::Ready => {
                // mark channel as closed
                self.channel
                    .state
                    .store(ChannelState::Closed as u8, Release);
            }
            // SAFETY: Receiver can be dropped in two situations:
            // - outside of `recv`, which means that the state is either Ready or Closed.
            // - in the `then` closure of `recv`, but there is a finally block that sets the state to
            // Ready before unwinding out of `recv`. Therefore, the state is either Ready or Closed.
            ChannelState::RecvWaiting | ChannelState::SenderWaiting => unsafe {
                std::hint::unreachable_unchecked()
            },
        }
    }
}

pub fn channel<T: Sync + Send>() -> (Sender<T>, Receiver<T>) {
    let channel = Arc::new(Channel {
        state: AtomicU8::new(ChannelState::Ready as u8),
        // dummy value: should never be read
        waiting_thread: std::panic::AssertUnwindSafe(UnsafeCell::new(thread::current())),
        message: std::panic::AssertUnwindSafe(UnsafeCell::new(ptr::null_mut())),
    });

    (
        Sender {
            channel: channel.clone(),
        },
        Receiver { channel },
    )
}

struct OnDrop<F: FnOnce()>(Option<F>);
impl<F: FnOnce()> Drop for OnDrop<F> {
    fn drop(&mut self) {
        if let Some(f) = self.0.take() {
            f()
        }
    }
}

fn finally<F: FnOnce()>(f: F) -> OnDrop<F> {
    OnDrop(Some(f))
}

#[cfg(test)]
mod test {
    use super::*;

    /// normal case
    #[test]
    fn send_recv_once() {
        let (mut sender, mut receiver) = channel();

        let t = thread::spawn(move || {
            while !sender.is_closed() {
                sender.send(&mut 42);
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
        });

        let result = receiver.recv(|&mut x| x).unwrap();

        assert_eq!(result, 42);

        drop(receiver);

        let _ = t.join();
    }

    // panic in `then`
    #[test]
    fn panic_while_recv() {
        let (mut sender, mut receiver) = channel();

        thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(100));
            loop {
                sender.send(&mut 42);
            }
        });

        let _ = std::panic::catch_unwind(AssertUnwindSafe(|| {
            receiver.recv(|&mut _x| panic!("test"));
        }));

        let result = receiver.recv(|&mut x| x).unwrap();

        assert_eq!(result, 42);
    }

    // drop while waiting for message
    #[test]
    fn drop_sender() {
        let (sender, mut receiver) = channel::<()>();

        thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(100));
            drop(sender);
        });

        let result = receiver.recv(|&mut x| x);

        assert!(result.is_none());
    }

    // drop before sending message
    #[test]
    fn drop_receiver_before_send() {
        let (sender, receiver) = channel::<()>();

        drop(receiver);

        assert!(sender.is_closed());
    }

    #[test]
    fn send_recv_loop() {
        let (mut sender, mut receiver) = channel();

        thread::spawn(move || {
            for mut i in 0..100 {
                sender.send(&mut i);
            }
        });

        let mut results = Vec::with_capacity(100);
        while let Some(result) = receiver.recv(|&mut x| x) {
            results.push(result);
        }

        dbg!(&results);
        assert!(results.windows(2).all(|w| w[0] < w[1]));
    }
}
