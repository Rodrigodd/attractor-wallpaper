package io.github.rodrigodd.attractorwallpaper

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.util.Log
import android.view.MotionEvent
import android.view.Surface
import android.view.SurfaceHolder
import android.view.SurfaceView
import androidx.preference.PreferenceManager

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


    private var nativeCtx: Long = 0

    // Returns a pointer to the native context
    private external fun nativeSurfaceCreated(surface: Surface): Long
    private external fun nativeSurfaceChanged(
        ctx: Long,
        surface: Surface,
        format: Int,
        width: Int,
        height: Int
    )

    private external fun nativeSurfaceDestroyed(ctx: Long, surface: Surface)

    private external fun nativeSurfaceRedrawNeeded(ctx: Long, surface: Surface)

    private external fun nativeUpdateConfigInt(ctx: Long, key: String, value: Int)

    private external fun nativeGetWallpaper(
        ctx: Long,
        bitmap: Bitmap,
        viewWidth: Int,
        viewHeight: Int
    ): Bitmap?

    fun getWallpaper(width: Int, height: Int, viewWidth: Int, viewHeight: Int): Bitmap? {
        if (nativeCtx == 0L) return null
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        return nativeGetWallpaper(nativeCtx, bitmap, viewWidth, viewHeight)
    }

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
        if (nativeCtx == 0L) return
        nativeSurfaceRedrawNeeded(nativeCtx, holder.surface)
    }

    var downTouch = false

    override fun onTouchEvent(event: MotionEvent): Boolean {
        super.onTouchEvent(event)

        // Listening for the down and up touch events.
        return when (event.action) {
            MotionEvent.ACTION_DOWN -> {
                downTouch = true
                true
            }

            MotionEvent.ACTION_UP -> if (downTouch) {
                val prefs = PreferenceManager.getDefaultSharedPreferences(context)

                Log.d(TAG, "onClick")
                if (nativeCtx != 0L) {
                    var seed = prefs.getString("seed", "0")?.toLong() ?: 0

                    if (event.x > width / 2) {
                        Log.d(TAG, "increment seed")
                        seed += 1
                    } else {
                        Log.d(TAG, "decrement seed")
                        seed -= 1
                    }

                    prefs.edit().putString("seed", seed.toString()).apply()
                }

                downTouch = false
                performClick()
                true
            } else {
                false
            }

            else -> false  // Return false for other touch events.
        }
    }


    override fun onAttachedToWindow() {
        super.onAttachedToWindow()

        val prefs = PreferenceManager.getDefaultSharedPreferences(context)

        prefs.registerOnSharedPreferenceChangeListener { pref, key ->
            Log.d(TAG, "onSharedPreferenceChanged: $key")
            if (nativeCtx == 0L) return@registerOnSharedPreferenceChangeListener

            when (key) {
                "multisampling" -> {
                    val value = pref.getInt(key, 1)
                    nativeUpdateConfigInt(nativeCtx, key, value)
                }

                "seed" -> {
                    val value = pref.getString("seed", "0")?.toLong() ?: 0
                    nativeUpdateConfigInt(nativeCtx, key, value.toInt())
                }

                "intensity" -> {
                    val value = pref.getInt(key, 100)
                    nativeUpdateConfigInt(nativeCtx, key, value)
                }

                "min_area" -> {
                    val value = pref.getInt(key, 25)
                    nativeUpdateConfigInt(nativeCtx, key, value)
                }
            }
        }
    }

    constructor(context: Context?) : super(context)
    constructor(context: Context?, attrs: AttributeSet?) : super(context, attrs)
    constructor(context: Context?, attrs: AttributeSet?, defStyleAttr: Int) : super(
        context,
        attrs,
        defStyleAttr
    )


}
