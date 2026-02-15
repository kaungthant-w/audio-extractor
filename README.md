# AI Music Transformer

![Application Screenshot](assets/screenshot.png)
![Application Screenshot](assets/screenshot_2.png)

A web application that separates vocals from music and synthesizes instruments using AI.

## Prerequisites

Before running the project, you need:

1.  **Node.js & npm** (v14+): [Download Node.js](https://nodejs.org/)
2.  **Python 3.8+**: [Download Python](https://www.python.org/)
    *   Ensure `python` and `pip` are in your PATH.
    *   Required libraries: `numpy`, `librosa`, `soundfile`, `torch`, `scipy`, `imageio-ffmpeg`, `python-minifier`.
3.  **PHP 7.4+**: [Download PHP](https://www.php.net/downloads)
    *   Ensure `php` is in your PATH.
4.  **FFmpeg**: [Download FFmpeg](https://ffmpeg.org/download.html)
    *   Ensure `ffmpeg` and `ffprobe` are in your PATH (for audio conversion).

## Installation

1.  **Install Node.js dependencies**:
    ```bash
    npm install
    ```
    This installs Tailwind CSS, PostCSS, and minification tools.

2.  **Install Python dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Development

To start working on the project, you need two terminal windows:

### 1. Run Tailwind CSS Watcher
This recompiles your CSS automatically whenever you change `index.html` or `input.css`.

```bash
npm run dev
```

### 2. Start PHP Server
Open a second terminal and run:

```bash
npm run serve
```
Or manually: `php -S localhost:8000`

Then open [http://localhost:8000](http://localhost:8000) in your browser.

## Production Build

To minify all files (HTML, CSS, PHP, Python) and create a production-ready `dist` folder:

```bash
npm run build
```
This runs `node minify.js`, which generates the `dist/` directory containing optimized files.

## Files
- `index.html`: Main frontend (links to `style.css`).
- `input.css`: Source CSS (Tailwind directives + custom styles).
- `style.css`: Generated CSS (output from `input.css`).
- `process.php`: Backend logic.
- `audio_engine.py`: AI processing logic.
- `minify.js`: Build script.
- `requirements.txt`: Python dependencies.
