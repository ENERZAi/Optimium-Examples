rootProject.name = "optimium-android"

pluginManagement {
    repositories {
        google()
        mavenCentral()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()

        maven {
            name = "local"
            url = uri("/Users/numver8638/Desktop/outputs/kotlin")
        }
    }
}