package io.github.rodrigodd.attractorwallpaper

import android.app.WallpaperManager
import android.content.SharedPreferences
import android.content.SharedPreferences.OnSharedPreferenceChangeListener
import android.graphics.drawable.Animatable
import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.content.res.AppCompatResources
import androidx.preference.EditTextPreference
import androidx.preference.ListPreference
import androidx.preference.Preference
import androidx.preference.PreferenceFragmentCompat
import androidx.preference.SeekBarPreference
import com.rarepebble.colorpicker.ColorPreference


class SettingsActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.settings_activity)

        val view = findViewById<AttractorSurfaceView>(R.id.surfaceView)!!

        if (savedInstanceState == null) {
            supportFragmentManager
                .beginTransaction()
                .replace(R.id.settings, SettingsFragment(view, this))
                .commit()
        }
        supportActionBar?.setDisplayHomeAsUpEnabled(true)
    }

    class SettingsFragment(
        private val surfaceView: AttractorSurfaceView,
        private val settingsActivity: SettingsActivity
    ) : PreferenceFragmentCompat(), OnSharedPreferenceChangeListener {
        companion object {
            private const val TAG = "SettingsActivity"
        }

        private var seedPref: EditTextPreference? = null
        private var intensityPref: SeekBarPreference? = null
        private var exponentPref: SeekBarPreference? = null
        private var minAreaPref: SeekBarPreference? = null
        private var setWallpaperPref: Preference? = null
        private var themePref: ListPreference? = null
        private var background_color1: ColorPreference? = null
        private var background_color2: ColorPreference? = null

        override fun onCreatePreferences(savedInstanceState: Bundle?, rootKey: String?) {
            setPreferencesFromResource(R.xml.preferences, rootKey)

            seedPref = findPreference("seed")!!
            intensityPref = findPreference("intensity")!!
            exponentPref = findPreference("exponent")!!
            minAreaPref = findPreference("min_area")!!
            setWallpaperPref = findPreference("set_wallpaper_button")!!
            themePref = findPreference("theme")!!
            background_color1 = findPreference("background_color1")!!
            background_color2 = findPreference("background_color2")!!

            themePref?.entries = AttractorSurfaceView.THEMES.keys.toTypedArray()
            themePref?.entryValues = themePref?.entries
        }

        override fun onPause() {
            super.onPause()
            preferenceManager
                .sharedPreferences?.unregisterOnSharedPreferenceChangeListener(this)
        }

        override fun onResume() {
            super.onResume()
            val prefs = preferenceManager.sharedPreferences ?: return

            for (key in prefs.all.keys) {
                onSharedPreferenceChanged(prefs, key)
            }

            prefs.registerOnSharedPreferenceChangeListener(this)

            setWallpaperPref?.setOnPreferenceClickListener {
                val icon =
                    AppCompatResources.getDrawable(
                        requireContext(),
                        R.drawable.loading_icon_anim
                    )

                if (icon != null && icon is Animatable) {
                    icon.start()
                    setWallpaperPref?.icon = icon
                }

                Thread {
                    val wallpaperManager = WallpaperManager.getInstance(requireContext())
                    var width = wallpaperManager.desiredMinimumWidth
                    var height = wallpaperManager.desiredMinimumHeight

                    val display = resources.displayMetrics
                    val displayWidth = display.widthPixels
                    val displayHeight = display.heightPixels

                    if (width <= 0 || height <= 0) {
                        Log.i(
                            TAG,
                            "desiredMinimumWidth or desiredMinimumHeight <= 0, fallback to display size"
                        )
                        // fallback to size of the default display
                        width = displayWidth
                        height = displayHeight
                    }

                    Log.i(TAG, "Getting wallpaper $width x $height")

                    val bitmap =
                        surfaceView.getWallpaper(height, width, displayWidth, displayHeight)
                    if (bitmap != null) wallpaperManager.setBitmap(bitmap)

                    settingsActivity.runOnUiThread {
                        setWallpaperPref?.icon = null
                    }
                }.start()

                return@setOnPreferenceClickListener true
            }
        }

        override fun onSharedPreferenceChanged(
            sharedPreferences: SharedPreferences?,
            key: String?
        ) {
            if (sharedPreferences == null || key == null) {
                Log.w(TAG, "Pref or Key is null: $sharedPreferences, $key");
                return;
            }
            when (key) {
                "seed" -> {
                    val seed = sharedPreferences.getString(key, "0")
                    seedPref?.text = seed
                    seedPref?.summary =
                        resources.getString(R.string.pref_seed_sum, seed)
                }

                "intensity" -> {
                    val intensity = sharedPreferences.getInt(key, 25)
                    intensityPref?.summary =
                        resources.getString(
                            R.string.pref_intensity_sum,
                            intensity.toFloat() / 100.0
                        )
                }

                "exponent" -> {
                    val exponent = sharedPreferences.getInt(key, 25)
                    exponentPref?.summary =
                        resources.getString(
                            R.string.pref_exponent_sum,
                            exponent.toFloat() / 100.0
                        )
                }

                "min_area" -> {
                    val minArea = sharedPreferences.getInt(key, 100)
                    minAreaPref?.summary =
                        resources.getString(
                            R.string.pref_min_area_sum,
                            minArea
                        )
                }

                "theme" -> {
                    val theme = sharedPreferences.getString(key, "Unknown")
                    themePref?.summary =
                        resources.getString(
                            R.string.pref_theme_sum,
                            theme
                        )
                }

                "background_color1" -> {
                    val color = sharedPreferences.getInt(key, 0)
                    background_color1?.color = color
                }

                "background_color2" -> {
                    val color = sharedPreferences.getInt(key, 0)
                    background_color2?.color = color
                }
            }
        }

        override fun onDisplayPreferenceDialog(preference: Preference) {
            if (preference is ColorPreference) {
                preference.showDialog(this, 0)
            } else {
                super.onDisplayPreferenceDialog(preference)
            }
        }
    }
}
