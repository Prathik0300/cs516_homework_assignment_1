# Moments

A photo sharing social networking app built with Python and Flask. The example application for the book _[Python Web Development with Flask (2nd edition)](https://helloflask.com/en/book/4)_ (《[Flask Web 开发实战（第 2 版）](https://helloflask.com/book/4)》).

Demo: http://moments.helloflask.com

![Screenshot](demo.png)

## Installation

Clone the repo:

```
$ git clone https://github.com/greyli/moments
$ cd moments
```

Install dependencies with [PDM](https://pdm.fming.dev):

```
$ pdm install
```

> [!TIP]
> If you don't have PDM installed, you can create a virtual environment with `venv` and install dependencies with `pip install -r requirements.txt`.

To initialize the app, run the `flask init-app` command:

```
$ pdm run flask init-app
```

If you just want to try it out, generate fake data with `flask lorem` command then run the app:

```
$ pdm run flask lorem
```

It will create a test account:

-   email: `admin@helloflask.com`
-   password: `moments`

Now you can run the app:

```
$ pdm run flask run
* Running on http://127.0.0.1:5000/
```

### Optional: Auto alt-text and image labels (Azure Vision)

This app can auto-generate image captions (alt text) and detect labels during upload using Azure AI Vision.

1. Create an Azure AI Vision resource and collect the endpoint and key.
2. Set environment variables before running the app:

```
export AZURE_VISION_ENDPOINT="https://<your-resource>.cognitiveservices.azure.com"
export AZURE_VISION_KEY="<your-key>"
```

If unset, uploads still work; captions/labels are simply skipped.

Generated data:

-   Photo.alt_text: caption for accessibility; used as `<img alt="...">`.
-   Photo.labels: comma-separated labels; included in full-text search.

Dependencies: `requests` (already included via requirements.txt).

## License

This project is licensed under the MIT License (see the
[LICENSE](LICENSE) file for details).
