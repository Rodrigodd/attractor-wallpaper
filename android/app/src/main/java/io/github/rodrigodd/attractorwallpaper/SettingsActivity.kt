package io.github.rodrigodd.attractorwallpaper

import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import androidx.preference.EditTextPreference
import androidx.preference.PreferenceFragmentCompat
import androidx.preference.SeekBarPreference

class SettingsActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.settings_activity)
        if (savedInstanceState == null) {
            supportFragmentManager
                .beginTransaction()
                .replace(R.id.settings, SettingsFragment())
                .commit()
        }
        supportActionBar?.setDisplayHomeAsUpEnabled(true)
    }

    class SettingsFragment : PreferenceFragmentCompat() {
        val TAG: String = "SettingsActivity"

        override fun onCreatePreferences(savedInstanceState: Bundle?, rootKey: String?) {
            setPreferencesFromResource(R.xml.preferences, rootKey)

            val prefs = preferenceManager.sharedPreferences ?: return

            val seedPref = findPreference<EditTextPreference>("seed")
            val intensityPref = findPreference<SeekBarPreference>("intensity")
            val minAreaPref = findPreference<SeekBarPreference>("min_area")
            if (seedPref == null || intensityPref == null || minAreaPref == null) {
                Log.e(TAG, "no pref found")
                return
            }


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
        }
    }
}
