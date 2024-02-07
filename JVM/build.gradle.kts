plugins {
    id("org.jetbrains.kotlin.jvm") version "1.8.10"
}

java {
    targetCompatibility = JavaVersion.VERSION_11
}

kotlin {
    jvmToolchain(11)
}

tasks.jar {
    manifest {
        attributes(
            "Main-Class" to "com.enerzai.optimium.example.ExampleKt"
        )
    }
}