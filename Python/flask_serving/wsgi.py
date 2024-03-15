import json

import numpy as np
import optimium.runtime as rt

from typing import Dict

from PIL import Image
from flask import Flask, jsonify, request, render_template


class ThisApp(Flask):
    labels: Dict[int, str]
    ctx: rt.Context
    model: rt.Model
    req: rt.InferRequest

    @staticmethod
    def __load_labels():
        # ImageNet labels
        with open("labels.json", "r", encoding="utf-8") as file:
            return {int(key): value for key, value in json.load(file).items()}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.labels = ThisApp.__load_labels()
        self.ctx = rt.Context()
        self.model = self.ctx.load_model("mobilenetv3")
        self.req = self.model.create_request()

    def fini(self):
        # del self.req
        # del self.model
        # del self.ctx
        pass


INPUT_SIZE = (224, 224)

app = ThisApp(__name__)


@app.route("/")
def main():
    return render_template("main.html")


def _preprocess_image(image: Image.Image) -> np.ndarray:
    data = np.array(image.resize(INPUT_SIZE)).astype(np.float32) / 127.5
    data -= 1.0

    return np.expand_dims(data, axis=0)


def _run_infer(data: np.ndarray) -> Dict[str, str]:
    app.req.set_inputs([data])

    app.req.infer()
    app.req.wait()

    output = app.req.get_outputs()[0]
    index = np.argmax(output)

    result = app.labels[index]

    return {"object": result[1], "id": result[0], "score": float(output[0, index])}


@app.route("/infer", methods=["POST"])
def run_inference():
    if app.req.status == rt.InferStatus.Running:
        return jsonify(
            {
                "status": "error",
                "cause": "other inference is already running; try again later.",
            }
        )

    try:
        file = request.files["file"]
        image = Image.open(file.stream)

        input_data = _preprocess_image(image)
        output = _run_infer(input_data)

        return jsonify({"status": "ok", "result": output})
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"result": "error", "cause": str(e)})


if __name__ == "__main__":
    app.run()
