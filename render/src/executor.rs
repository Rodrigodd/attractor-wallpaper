use super::UserEvent;
use std::{
    collections::HashMap,
    future::Future,
    pin::Pin,
    sync::Mutex,
    task::{Context, Poll},
};
use winit::event_loop::EventLoopProxy;

pub type TaskId = usize;

pub struct WinitExecutor {
    tasks: HashMap<TaskId, Pin<Box<dyn Future<Output = ()>>>>,
    event_loop_proxy: EventLoopProxy<UserEvent>,
}
impl WinitExecutor {
    /// Create a new `WinitExecutor`, driven by the given event loop.
    pub fn new(event_loop_proxy: EventLoopProxy<UserEvent>) -> Self {
        Self {
            tasks: HashMap::new(),
            event_loop_proxy,
        }
    }

    fn next_task_id(&self) -> TaskId {
        static NEXT_TASK_ID: std::sync::atomic::AtomicUsize =
            std::sync::atomic::AtomicUsize::new(0);
        NEXT_TASK_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    }

    /// Spawn a task.
    ///
    /// This immediately pools the task once, and then schedules it to be
    /// polled again if needed, using a `UserEvent::PollTask` event.
    pub fn spawn(&mut self, task: impl Future<Output = ()> + 'static) {
        let task = Box::pin(task);
        let task_id = self.next_task_id();
        self.tasks.insert(task_id, task);
        self.poll(task_id);
    }

    /// Poll a task.
    ///
    /// Should be called when the event loop receives a `UserEvent::PollTask`.
    pub fn poll(&mut self, task_id: TaskId) {
        log::trace!("polling task {}", task_id);
        let winit_proxy = Mutex::new(self.event_loop_proxy.clone());
        let waker = waker_fn::waker_fn(move || {
            log::trace!("waking task {}", task_id);
            let _ = winit_proxy
                .lock()
                .unwrap()
                .send_event(UserEvent::PollTask(task_id));
        });
        let task = self.tasks.get_mut(&task_id).unwrap().as_mut();
        match task.poll(&mut Context::from_waker(&waker)) {
            Poll::Ready(()) => {
                self.tasks.remove(&task_id);
            }
            Poll::Pending => {}
        }
    }
}
