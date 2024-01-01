package io.github.rodrigodd.attractorwallpaper

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import androidx.preference.EditTextPreference
import androidx.preference.PreferenceFragmentCompat

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
        override fun onCreatePreferences(savedInstanceState: Bundle?, rootKey: String?) {
            setPreferencesFromResource(R.xml.preferences, rootKey)

            val prefs = preferenceManager.sharedPreferences
            prefs?.registerOnSharedPreferenceChangeListener { sharedPreferences, key ->
                when (key) {
                    "seed" -> {
                        val seed = sharedPreferences.getString(key, "0")
                        val pref = findPreference<EditTextPreference>("seed")
                        pref?.text = seed
                    }
                }
            }
        }
    }
}
