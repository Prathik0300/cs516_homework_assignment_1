import os
from pathlib import Path

from dotenv import load_dotenv

dotenv_path = Path(__file__).resolve().parent / '.env'
if dotenv_path.exists():
    loaded = load_dotenv(dotenv_path)
    # Debug: verify .env is loaded and Azure vars are present
    print(f"[dotenv] loaded={loaded} path={dotenv_path}")
    print(f"[dotenv] AZURE_VISION_ENDPOINT={os.getenv('AZURE_VISION_ENDPOINT')}")
    print(f"[dotenv] AZURE_VISION_KEY set={os.getenv('AZURE_VISION_KEY')}")

from moments import create_app  # noqa

config_name = os.getenv('FLASK_CONFIG', 'development')
app = create_app(config_name)
