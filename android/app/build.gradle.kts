plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
}

android {
    namespace = "com.muzi.mini"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.muzi.mini"
        minSdk = 26            // Moto G60 (Android 11+) folgado
        targetSdk = 34
        versionCode = 1
        versionName = "0.1"
        // Moto G60 = arm64. Mantemos so arm64 pra reduzir o APK.
        ndk { abiFilters += listOf("arm64-v8a") }
    }
    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
    buildFeatures { compose = true }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.4")
    implementation("androidx.activity:activity-compose:1.9.1")
    implementation(platform("androidx.compose:compose-bom:2024.08.00"))
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.8.4")

    // Motor de inferencia: ONNX Runtime core para Android (CPU-only, NEON).
    // Resolve limpo no Maven. O loop de geracao (KV cache) e feito por cima.
    // Obs: onnxruntime-genai-android NAO esta no Maven como dep simples —
    // precisa ser compilado do fonte; por isso usamos o core aqui.
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.26.0")

    // Download de modelos
    implementation("com.squareup.okhttp3:okhttp:4.12.0")
}
