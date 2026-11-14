import uvicorn
import os
from pathlib import Path

if __name__ == "__main__":
    # Локальный запуск вне Docker (для разработки).
    # В Docker backend стартует командой uvicorn из Dockerfile.backend (без SSL).
    kwargs = {}
    cert_dir = Path(__file__).resolve().parent / "certs"
    key_file, cert_file = cert_dir / "key.pem", cert_dir / "cert.pem"
    if key_file.exists() and cert_file.exists():
        kwargs["ssl_keyfile"] = str(key_file)
        kwargs["ssl_certfile"] = str(cert_file)

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("BACKEND_PORT", "8002")),
        reload=True,
        **kwargs,
    )