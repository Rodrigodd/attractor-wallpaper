package io.github.rodrigodd.attractorwallpaper

import android.content.Context
import android.content.SharedPreferences
import android.content.SharedPreferences.OnSharedPreferenceChangeListener
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
import java.nio.ByteBuffer

class AttractorSurfaceView : SurfaceView, SurfaceHolder.Callback2,
    OnSharedPreferenceChangeListener {

    class Theme(
        val background_color1: Int,
        val background_color2: Int,
        val gradient: Array<Pair<Float, Int>>,
    );

    init {
        holder.addCallback(this)
    }

    companion object {
        private const val TAG = "AttractorSurfaceView"

        private val HOT_THEME: Theme = Theme(
            background_color1 = Color.rgb(25, 0, 0),
            background_color2 = Color.rgb(40, 0, 0),
            gradient = arrayOf(
                Pair(0.00f, Color.rgb(7, 1, 0)),
                Pair(0.03f, Color.rgb(71, 0, 0)),
                Pair(0.30f, Color.rgb(188, 0, 1)),
                Pair(0.75f, Color.rgb(249, 234, 0)),
                Pair(1.00f, Color.rgb(255, 255, 255))
            )
        )

        private val FOREST_THEME: Theme = Theme(
            background_color1 = Color.rgb(0, 35, 0),
            background_color2 = Color.rgb(9, 20, 0),
            gradient = arrayOf(
                Pair(0.00f, Color.rgb(1, 3, 8)),
                Pair(0.02f, Color.rgb(2, 39, 34)),
                Pair(0.30f, Color.rgb(74, 109, 60)),
                Pair(0.75f, Color.rgb(208, 240, 195)),
                Pair(1.00f, Color.rgb(91, 250, 0))
            ),
        )

        private val COLD_THEME: Theme = Theme(
            background_color1 = Color.rgb(0, 20, 40),
            background_color2 = Color.rgb(0, 10, 25),
            gradient = arrayOf(
                Pair(0.000f, Color.rgb(1, 3, 8)),
                Pair(0.025f, Color.rgb(2, 38, 47)),
                Pair(0.300f, Color.rgb(50, 103, 144)),
                Pair(0.750f, Color.rgb(175, 241, 255)),
                Pair(1.000f, Color.rgb(255, 253, 247))
            )
        )

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

    private external fun nativeUpdateTheme(ctx: Long, theme: ByteBuffer);

    fun serializeTheme(theme: Theme): ByteBuffer {
        var buffer = ByteBuffer.allocateDirect(4 * 5 * 2 + 4 * 2)
        buffer.order(java.nio.ByteOrder.LITTLE_ENDIAN)
        buffer.putInt(theme.background_color1)
        buffer.putInt(theme.background_color2)
        for (pair in theme.gradient) {
            buffer.putFloat(pair.first)
            buffer.putInt(pair.second)
        }

        buffer.flip()
        return buffer
    }

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
            return
        }

        val prefs = PreferenceManager.getDefaultSharedPreferences(context)
        prefs.registerOnSharedPreferenceChangeListener(this)

        for (key in prefs.all.keys) {
            onSharedPreferenceChanged(prefs, key)
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

        val pref = PreferenceManager.getDefaultSharedPreferences(context)
        pref.unregisterOnSharedPreferenceChangeListener(this)
    }

    override fun surfaceRedrawNeeded(holder: SurfaceHolder) {
        Log.d(TAG, "surfaceRedrawNeeded")
        if (nativeCtx == 0L) return
        nativeSurfaceRedrawNeeded(nativeCtx, holder.surface)
    }

    override fun onSharedPreferenceChanged(pref: SharedPreferences, key: String) {
        Log.d(TAG, "onSharedPreferenceChanged: $key")
        if (nativeCtx == 0L) return

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

            "exponent" -> {
                val value = pref.getInt(key, 100)
                nativeUpdateConfigInt(nativeCtx, key, value)
            }

            "min_area" -> {
                val value = pref.getInt(key, 25)
                nativeUpdateConfigInt(nativeCtx, key, value)
            }

            "background_color1" -> {
                val value = pref.getInt(key, Color.BLACK)
                nativeUpdateConfigInt(nativeCtx, key, value)
                Log.i(TAG, "background_color1: $value")
            }

            "background_color2" -> {
                val value = pref.getInt(key, Color.BLACK)
                nativeUpdateConfigInt(nativeCtx, key, value)
            }

            "theme" -> {
                val value = pref.getString(key, "Unknown")
                val theme = when (value) {
                    "Hot" -> HOT_THEME
                    "Forest" -> FOREST_THEME
                    "Cold" -> COLD_THEME
                    else -> {
                        Log.e(TAG, "Unknown theme: $value")
                        return
                    }
                }

                nativeUpdateTheme(nativeCtx, serializeTheme(theme))
                pref.edit().let {
                    it.putInt("background_color1", theme.background_color1)
                    it.putInt("background_color2", theme.background_color2)
                    it.apply()
                }
            }
        }
    }

    private var downTouch = false

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

    override fun performClick(): Boolean {
        super.performClick()
        return true
    }

    constructor(context: Context?) : super(context)
    constructor(context: Context?, attrs: AttributeSet?) : super(context, attrs)
    constructor(context: Context?, attrs: AttributeSet?, defStyleAttr: Int) : super(
        context,
        attrs,
        defStyleAttr
    )
}
