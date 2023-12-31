package io.github.rodrigodd.attractorwallpaper

import android.content.Context
import android.graphics.BlendMode
import android.graphics.Color
import android.graphics.Paint
import android.graphics.PorterDuff
import android.util.AttributeSet
import android.util.Log
import android.view.Surface
import android.view.SurfaceHolder
import android.view.SurfaceView

class AttractorSurfaceView : SurfaceView, SurfaceHolder.Callback2 {
    init {
        holder.addCallback(this)
    }

    companion object {
        private const val TAG = "AttractorSurfaceView"

        init {
            try {
                System.loadLibrary("attractor_android")
            } catch (e: UnsatisfiedLinkError) {
                Log.e(TAG, "Failed to load native library", e)
            }
        }
    }


    private var nativeCtx: Long = 0;

    // Returns a pointer to the native context
    private external fun nativeSurfaceCreated(surface: Surface): Long;
    private external fun nativeSurfaceChanged(
        ctx: Long,
        surface: Surface,
        format: Int,
        width: Int,
        height: Int
    );

    private external fun nativeSurfaceDestroyed(ctx: Long, surface: Surface);

    override fun surfaceCreated(holder: SurfaceHolder) {
        Log.i("AttractorSurfaceView", "surfaceCreated")

        // setBackgroundColor(Color.BLUE)

        val surface = holder.surface
        nativeCtx = nativeSurfaceCreated(surface)

        // draw text "context creation failed" if nativeCtx == 0L
        if (nativeCtx == 0L) {
            Log.e(TAG, "context creation failed")
            setBackgroundColor(Color.RED)
            val canvas = surface.lockHardwareCanvas()
            canvas.drawText("context creation failed", 0f, 0f, Paint().apply {
                color = Color.WHITE
                textSize = 100f
            })
            surface.unlockCanvasAndPost(canvas)
        }
    }

    override fun surfaceChanged(holder: SurfaceHolder, format: Int, width: Int, height: Int) {
        Log.i(
            TAG,
            "surfaceChanged: $nativeCtx, $holder, $format $width, $height"
        )
        if (nativeCtx == 0L) return
        nativeSurfaceChanged(nativeCtx, holder.surface, format, width, height)
    }

    override fun surfaceDestroyed(holder: SurfaceHolder) {
        Log.i(TAG, "surfaceDestroy")
        if (nativeCtx == 0L) return
        nativeSurfaceDestroyed(nativeCtx, holder.surface)
    }

    override fun surfaceRedrawNeeded(holder: SurfaceHolder) {
        Log.d(TAG, "surfaceRedrawNeeded")
        //setBackgroundColors(Color.BLUE)
    }

    constructor(context: Context?) : super(context)
    constructor(context: Context?, attrs: AttributeSet?) : super(context, attrs)
    constructor(context: Context?, attrs: AttributeSet?, defStyleAttr: Int) : super(
        context,
        attrs,
        defStyleAttr
    )

    constructor(
        context: Context?,
        attrs: AttributeSet?,
        defStyleAttr: Int,
        defStyleRes: Int
    ) : super(context, attrs, defStyleAttr, defStyleRes)


}
