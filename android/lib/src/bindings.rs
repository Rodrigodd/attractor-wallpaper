//! JNI bindings to native methods of io.github.rodrigodd.attractorwallpaper.AttractorSurfaceView
#![allow(non_snake_case)]

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
        super::on_resize(&*ctx, width, height);
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
        super::on_redraw(&*ctx);
    }
}

/// #nativeUpdateConfigInt(ctx: Long, key: String, value: Int)
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
        super::on_update_config_int(&*ctx, key, value);
    }
}

/// #nativeGetWallpaper(ctx: Long, bitmap: Bitmap, viewWidth: Int, viewHeight: Int): Bitmap?
#[no_mangle]
fn Java_io_github_rodrigodd_attractorwallpaper_AttractorSurfaceView_nativeGetWallpaper<'a>(
    env: JNIEnv,
    _: JClass,
    ctx: jlong,
    bitmap_obj: JObject<'a>,
    view_width: u32,
    view_height: u32,
) -> JObject<'a> {
    let ctx = ctx as *mut super::Context;
    log::debug!("nativeGetWallpaper: {:p} {:?}", ctx, bitmap_obj);

    let bitmap =
        unsafe { ndk::bitmap::Bitmap::from_jni(env.get_native_interface(), bitmap_obj.as_raw()) };

    let Ok(info) = bitmap.info() else {
        log::error!("Failed to get bitmap info");
        return JObject::null();
    };
    let width = info.width();
    let height = info.height();

    if info.format() != ndk::bitmap::BitmapFormat::RGBA_8888 {
        log::error!("Bitmap format is not RGBA_8888");
        return JObject::null();
    }

    let Ok(pixels) = bitmap.lock_pixels() else {
        log::error!("Failed to lock bitmap pixels");
        return JObject::null();
    };

    let pixels =
        unsafe { std::slice::from_raw_parts_mut(pixels as *mut u8, (width * height * 4) as usize) };

    unsafe {
        super::on_get_wallpaper(&*ctx, width, height, view_width, view_height, pixels);
    }

    let _ = bitmap.unlock_pixels();

    bitmap_obj
}

// #nativeUpdateTheme(ctx: Long, theme: ByteBuffer)
#[no_mangle]
fn Java_io_github_rodrigodd_attractorwallpaper_AttractorSurfaceView_nativeUpdateTheme(
    env: JNIEnv,
    _: JClass,
    ctx: jlong,
    theme: jni::objects::JByteBuffer,
) {
    let ctx = ctx as *mut super::Context;
    log::debug!("nativeUpdateTheme: {:p} {:?}", ctx, theme);

    let ptr = env.get_direct_buffer_address(&theme);
    let len = env.get_direct_buffer_capacity(&theme);

    let (ptr, len) = match (ptr, len) {
        (Ok(ptr), Ok(len)) => {
            log::debug!("theme: {:p} {}", ptr, len);
            (ptr, len)
        }
        err => {
            log::error!("Failed to get theme pointer/length: {:?}", err);
            return;
        }
    };

    let theme = unsafe { std::slice::from_raw_parts(ptr as *const u8, len as usize) };

    unsafe {
        super::on_update_theme(&*ctx, theme);
    }
}
