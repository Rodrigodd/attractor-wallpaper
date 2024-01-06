package io.github.rodrigodd.attractorwallpaper

import android.app.WallpaperManager
import android.os.Bundle
import android.util.Log
import android.view.WindowManager
import android.view.WindowMetrics
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.content.res.AppCompatResources
import androidx.preference.EditTextPreference
import androidx.preference.Preference
import androidx.preference.PreferenceFragmentCompat
import androidx.preference.SeekBarPreference

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
        val surfaceView: AttractorSurfaceView,
        val settingsActivity: SettingsActivity
    ) : PreferenceFragmentCompat() {
        val TAG: String = "SettingsActivity"

        override fun onCreatePreferences(savedInstanceState: Bundle?, rootKey: String?) {
            setPreferencesFromResource(R.xml.preferences, rootKey)

            val prefs = preferenceManager.sharedPreferences ?: return

            val seedPref = findPreference<EditTextPreference>("seed")!!
            val intensityPref = findPreference<SeekBarPreference>("intensity")!!
            val minAreaPref = findPreference<SeekBarPreference>("min_area")!!
            val setWallpaperPref = findPreference<Preference>("set_wallpaper_button")!!

            seedPref.summary =
                resources.getString(R.string.pref_seed_sum, prefs.getString("seed", "0"))
            intensityPref.summary =
                resources.getString(
                    R.string.pref_intensity_sum,
                    prefs.getInt("intensity", 100).toFloat() / 100.0
                )
            minAreaPref.summary =
                resources.getString(
                    R.string.pref_min_area_sum,
                    prefs.getInt("min_area", 25)
                )

            prefs.registerOnSharedPreferenceChangeListener { sharedPreferences, key ->
                when (key) {
                    "seed" -> {
                        val seed = sharedPreferences.getString(key, "0")
                        seedPref.text = seed
                        seedPref.summary =
                            resources.getString(R.string.pref_seed_sum, seed)
                    }

                    "intensity" -> {
                        val intensity = sharedPreferences.getInt(key, 25)
                        intensityPref.summary =
                            resources.getString(
                                R.string.pref_intensity_sum,
                                intensity.toFloat() / 100.0
                            )
                    }

                    "min_area" -> {
                        val min_area = sharedPreferences.getInt(key, 100)
                        minAreaPref.summary =
                            resources.getString(
                                R.string.pref_min_area_sum,
                                min_area
                            )
                    }
                }
            }

            setWallpaperPref.setOnPreferenceClickListener {
                val icon =
                    AppCompatResources.getDrawable(
                        requireContext(),
                        R.drawable.ic_launcher_foreground
                    )
                if (icon != null) setWallpaperPref.icon = icon

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
                        setWallpaperPref.icon = null
                    }
                }.start()

                return@setOnPreferenceClickListener true
            }
        }
    }
}
