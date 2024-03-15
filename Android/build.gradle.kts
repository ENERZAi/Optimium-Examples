plugins {
    id("org.jetbrains.kotlin.android") version "1.8.10"
    id("com.android.application") version "8.2.0"
}

android {
    namespace = "com.enerzai.optimium.example.android"
    compileSdk = 33

    defaultConfig {
        applicationId = "com.enerzai.optimium.example.android"
        minSdk = 26
        targetSdk = 33
        versionCode = 1
        versionName = "1.0"

        ndk {
            //noinspection ChromeOsAbiSupport
            abiFilters += setOf("x86_64", "armeabi-v7a", "arm64-v8a")
        }
    }
    buildTypes {
        release {
            isMinifyEnabled = false
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
        buildConfig = true
    }
    composeOptions {
        kotlinCompilerExtensionVersion = "1.4.4"
    }
    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
        jniLibs {
            arrayOf("x86_64", "armeabi-v7a", "arm64-v8a").forEach {
                pickFirsts.add("lib/$it/libc++_shared.so")
            }
        }
    }
}

dependencies {
    val core_version = "1.9.0"
    implementation("androidx.core:core-ktx:${core_version}")

    val activity_version = "1.7.2"
    implementation("androidx.activity:activity-compose:${activity_version}")

    val compose_version = "1.4.3"
    val compose_bom_version = "2023.06.01" // should be matched with compose_version
    implementation("androidx.compose.ui:ui:${compose_version}")
    implementation("androidx.compose.ui:ui-graphics:${compose_version}")
    implementation("androidx.compose.ui:ui-tooling-preview:${compose_version}")
    implementation("androidx.compose.material:material:${compose_version}")
    implementation(platform("androidx.compose:compose-bom:${compose_bom_version}"))
    debugImplementation("androidx.compose.ui:ui-tooling:${compose_version}")

    val lifecycle_version = "2.6.2"
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:${lifecycle_version}")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:${lifecycle_version}")

    val camerax_version = "1.2.3"
    implementation("androidx.camera:camera-core:${camerax_version}")
    implementation("androidx.camera:camera-camera2:${camerax_version}")
    implementation("androidx.camera:camera-lifecycle:${camerax_version}")
    implementation("androidx.camera:camera-view:${camerax_version}")

    val exif_interface_version = "1.3.7"
    implementation("androidx.exifinterface:exifinterface:${exif_interface_version}")

    implementation("com.enerzai.optimium.runtime:android:0.3.2")
    runtimeOnly("com.enerzai.optimium.runtime:android-native:0.3.2")
    runtimeOnly("com.enerzai.optimium.runtime:android-xnnpack:0.3.2")

    implementation(files("libs/opencv_java_shared_4.9.0.aar"))
}