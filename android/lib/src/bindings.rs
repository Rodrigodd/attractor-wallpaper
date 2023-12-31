//! JNI bindings to native methods of io.github.rodrigodd.attractorwallpaper.AttractorSurfaceView

use jni::{
    objects::{JClass, JObject, JString},
    sys::{jint, jlong},
    JNIEnv,
};

/// #nativeSurfaceCreated(surface: Surface): Long
#[no_mangle]
pub extern "system" fn Java_io_github_rodrigodd_attractorwallpaper_AttractorSurfaceView_nativeSurfaceCreated(
    env: JNIEnv,
    _: JClass,
    surface: JObject,
) -> jlong {
    super::init();
    log::debug!("nativeSurfaceCreated: {:?}", surface);

    let window = unsafe {
        ndk::native_window::NativeWindow::from_surface(env.get_native_interface(), surface.as_raw())
    };

    let Some(window) = window else {
        return 0;
    };

    let ctx = super::on_create(super::NativeWindow(window));

    let ctx = match ctx {
        Ok(ctx) => ctx,
        Err(err) => {
            log::error!("Failed to create context: {:?}", err);
            return 0;
        }
    };

    let ptr = Box::into_raw(ctx);

    log::debug!("nativeSurfaceCreated: {:p} {:?}", ptr, surface);

    ptr as usize as i64
}

/// #nativeSurfaceChanged(ctx: Long, surface: Surface, format: int, width: int, height: int)
#[no_mangle]
pub extern "system" fn Java_io_github_rodrigodd_attractorwallpaper_AttractorSurfaceView_nativeSurfaceChanged(
    _env: JNIEnv,
    _: JClass,
    ctx: jlong,
    surface: JObject,
    format: jint,
    width: jint,
    height: jint,
) {
    let ctx = ctx as *mut super::Context;
    log::debug!(
        "nativeSurfaceChanged: {:p} {:?} {} {} {}",
        ctx,
        surface,
        format,
        width,
        height
    );
    unsafe {
        super::on_resize(&mut *ctx, width, height);
    }
}

/// #nativeSurfaceDestroyed(ctx: Long, surface: Surface)
#[no_mangle]
pub extern "system" fn Java_io_github_rodrigodd_attractorwallpaper_AttractorSurfaceView_nativeSurfaceDestroyed(
    _env: JNIEnv,
    _: JClass,
    ctx: jlong,
    surface: JObject,
) {
    let ctx = ctx as *mut super::Context;
    log::debug!("nativeSurfaceDestroyed: {:p} {:?}", ctx, surface,);
    unsafe {
        super::on_destroy(Box::from_raw(ctx));
    }
}

/// #nativeSurfaceRedrawNeeded(ctx: Long, surface: Surface);
#[no_mangle]
pub extern "system" fn Java_io_github_rodrigodd_attractorwallpaper_AttractorSurfaceView_nativeSurfaceRedrawNeeded(
    _env: JNIEnv,
    _: JClass,
    ctx: jlong,
    surface: JObject,
) {
    let ctx = ctx as *mut super::Context;
    log::debug!("nativeSurfaceRedrawNeeded: {:p} {:?}", ctx, surface,);
    unsafe {
        super::on_redraw(&mut *ctx);
    }
}

/// #fun nativeUpdateConfigInt(ctx: Long, key: String, value: Int)
#[no_mangle]
pub extern "system" fn Java_io_github_rodrigodd_attractorwallpaper_AttractorSurfaceView_nativeUpdateConfigInt(
    _env: JNIEnv,
    _: JClass,
    ctx: jlong,
    key: JString,
    value: jint,
) {
    let ctx = ctx as *mut super::Context;
    // SAFETY: I control the other side of the FFI boundary, so I can guarantee
    // that key is a valid string.
    let key = unsafe { _env.get_string_unchecked(&key) };
    let Ok(key) = key else {
        log::error!("key is null");
        return;
    };
    let Ok(key) = key.to_str() else {
        log::error!("key is not valid UTF-8");
        return;
    };
    log::debug!("nativeUpdateConfigInt: {:p} {} {}", ctx, key, value);
    unsafe {
        super::on_update_config_int(&mut *ctx, key, value);
    }
}
