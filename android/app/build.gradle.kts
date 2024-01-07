plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.mozilla.rust-android-gradle.rust-android") version "0.9.3"
    id("com.dorongold.task-tree") version "2.1.1"
}

android {
    namespace = "io.github.rodrigodd.attractorwallpaper"
    compileSdk = 33

    ndkVersion = "24.0.8215888"

    defaultConfig {
        applicationId = "io.github.rodrigodd.attractorwallpaper"
        minSdk = 23
        targetSdk = 33
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        vectorDrawables {
            useSupportLibrary = true
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
    }
    kotlinOptions {
        jvmTarget = "1.8"
    }
    buildFeatures {
        compose = true
    }
    composeOptions {
        kotlinCompilerExtensionVersion = "1.4.3"
    }
    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
    }
}

cargo {
    module = "../lib"
    libname = "attractor_android"
    // targets = listOf("x86_64")
    targets = listOf("arm64")
    targetDirectory = "../../target"
    profile = "release"
}

tasks.whenTaskAdded {
    if (name == "mergeDebugJniLibFolders" || name == "mergeReleaseJniLibFolders") {
        dependsOn("cargoBuild")
        inputs.dir(buildDir.resolve("rustJniLibs/android").apply { mkdirs() })
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.9.0")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.6.1")
    implementation("androidx.activity:activity-compose:1.7.0")
    implementation(platform("androidx.compose:compose-bom:2023.03.00"))
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-graphics")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.preference:preference-ktx:1.2.1")
    implementation("androidx.appcompat:appcompat:1.6.1")
    implementation("com.google.android.material:material:1.4.+")
    implementation("com.github.martin-stone:hsv-alpha-color-picker-android:3.1.0")
    debugImplementation("androidx.compose.ui:ui-test-manifest")
    debugImplementation("androidx.compose.ui:ui-tooling")
    androidTestImplementation("androidx.compose.ui:ui-test-junit4")
    androidTestImplementation(platform("androidx.compose:compose-bom:2023.03.00"))
    androidTestImplementation("androidx.test.espresso:espresso-core:3.5.1")
    androidTestImplementation("androidx.test.ext:junit:1.1.5")
    testImplementation("junit:junit:4.13.2")
}
