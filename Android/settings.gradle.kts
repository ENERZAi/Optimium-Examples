rootProject.name = "optimium-android"

pluginManagement {
    repositories {
        google()
        mavenCentral()
    }
}

val OPTIMIUM_SDK_ROOT = System.getenv("OPTIMIUM_SDK_ROOT") ?: throw GradleException("OPTIMIUM_SDK_ROOT is not set.")

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()

        maven {
            name = "local"
            url = uri("$OPTIMIUM_SDK_ROOT/kotlin")
        }
    }
}