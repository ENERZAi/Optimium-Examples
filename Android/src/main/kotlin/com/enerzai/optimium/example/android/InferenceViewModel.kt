package com.enerzai.optimium.example.android

import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import android.provider.MediaStore
import android.util.Log
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.core.content.FileProvider
import androidx.exifinterface.media.ExifInterface
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.enerzai.optimium.runtime.ContextFactory
import com.enerzai.optimium.runtime.logging.LogLevel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.opencv.core.Mat
import org.opencv.core.Scalar
import java.io.File
import java.io.FileOutputStream
import kotlin.system.measureNanoTime

enum class UiState {
    INIT,
    WAIT,
    DATA_SELECTION,
    INFER,
    DONE,
    ERROR
}

class InferenceViewModel(private val appContext: Context) : ViewModel() {
    companion object {

        private val RED = Scalar(1.0, 0.0, 0.0)
        private const val THICKNESS = 2

        private const val FILE_PROVIDER = BuildConfig.APPLICATION_ID + ".file-provider"
        private const val MODEL_NAME = "model" // edit it to your model name
    }

    var uiState: UiState by mutableStateOf(UiState.INIT)
        private set

    init {
        init()
    }

    lateinit var imagePath: Uri

    private lateinit var modelPath: File
    private lateinit var context: com.enerzai.optimium.runtime.Context
    private lateinit var model: com.enerzai.optimium.runtime.Model
    private lateinit var request: com.enerzai.optimium.runtime.InferRequest

    private lateinit var anchors: List<FloatPair>
    private lateinit var input: FloatArray // 1 * 128 * 128 * 3
    private lateinit var classficators: FloatArray // 1 * 896 * 1
    private lateinit var regressors: FloatArray // 1 * 896 * 16

    var outputImage: Bitmap = Bitmap.createBitmap(300, 300, Bitmap.Config.ARGB_8888)
        private set

    var results: List<Detection> = emptyList()
        private set

    var stats: Map<String, Float> = emptyMap()

    var cause: String? = null
        private set

    private lateinit var image: Mat

    private fun init() {
        viewModelScope.launch(Dispatchers.IO) {
            val state = try {
                appContext.filesDir.mkdirs()

                // Extract model in temporary folder
                modelPath = extract(MODEL_NAME)

                // Create context
                val factory = ContextFactory()
                factory.enableLogcat().verbosity(LogLevel.INFO)

                context = factory.create()

                // Loading model
                model = context.loadModel(modelPath)

                // Create request
                request = model.createRequest()

                // Prepare buffer
                val inputInfo = model.getTensorInfo("input")
                val classificatorsInfo = model.getTensorInfo("classificators")
                val regressorsInfo = model.getTensorInfo("regressors")

                input = FloatArray(inputInfo.shape.totalElementCount.toInt())
                classficators = FloatArray(classificatorsInfo.shape.totalElementCount.toInt())
                regressors = FloatArray(regressorsInfo.shape.totalElementCount.toInt())

                addCloseable {
                    request.close()
                    model.close()
                    context.close()
                }

                anchors = calculateAnchors()

                UiState.WAIT
            } catch (ex: Exception) {
                Log.d(TAG, "failed to load a model", ex)
                cause = ex.message
                UiState.ERROR
            }

            withContext(Dispatchers.Main) {
                uiState = state
            }
        }
    }

    fun selectData() {
        uiState = UiState.DATA_SELECTION
    }

    fun cancel() {
        uiState = UiState.WAIT
    }

    fun infer(uri: Uri) {
        uiState = UiState.WAIT

        val exif = appContext.contentResolver.openInputStream(uri)?.let { ExifInterface(it) }
            ?: throw IllegalArgumentException("unexpected null")

        val orientation = exif.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_UNDEFINED)
        val bitmap = MediaStore.Images.Media.getBitmap(appContext.contentResolver, uri)

        doInfer(bitmap, orientation == ExifInterface.ORIENTATION_ROTATE_90)
    }

    fun createTempImage() {
        val dir = File(appContext.cacheDir, "inputs")
        dir.mkdirs()

        val file = File.createTempFile("INPUT_", ".jpg", dir)
        imagePath = FileProvider.getUriForFile(appContext, FILE_PROVIDER, file)
    }

    private fun extract(name: String): File {
        val assets = appContext.assets
        val baseDir = File(appContext.codeCacheDir, name)

        assets.list(name)?.forEach {
            val outputFile = File(baseDir, it)

            outputFile.parentFile?.mkdirs()

            assets.open("$name/$it").use { input ->
                FileOutputStream(outputFile).use { output ->
                    input.copyTo(output)
                }
            }
        }

        return baseDir
    }

    private fun doInfer(data: Bitmap, rotated: Boolean) {
        viewModelScope.launch(Dispatchers.IO) {
            val image: Mat
            val roi: FloatPair
            val preprocess = measureNanoTime {
                image = resizeData(data, rotated)
                roi = preprocessData(image, input)
            }

            val input = measureNanoTime {
                request.setInput("input", input)
            }

            val infer = measureNanoTime {
                request.infer()
                request.waitForFinish()
            }

            val output = measureNanoTime {
                request.getOutput("classificators", classficators)
                request.getOutput("regressors", regressors)
            }

            val postprocess = measureNanoTime {
                results = postprocessData(regressors, classficators, anchors, roi)

                // Create new bitmap if image size does not match. Otherwise reuse it.
                if (outputImage.width != image.width() || outputImage.height != image.height()) {
                    outputImage =
                        Bitmap.createBitmap(image.width(), image.height(), Bitmap.Config.ARGB_8888)
                }

                drawDetections(image, results, outputImage, color = RED, thickness = THICKNESS)
            }

            image.release()

            stats = mapOf(
                "Preprocess" to preprocess / 1000f,
                "Input" to input / 1000f,
                "Infer" to infer / 1000f,
                "Output" to output / 1000f,
                "PostProcess" to postprocess / 1000f
            )

            uiState = UiState.DONE
        }
    }
}