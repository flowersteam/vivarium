"""
Ngrok tunnel management and Google Colab environment detection.

Provides functions to create/close ngrok tunnels for exposing
local services publicly, and to validate Colab environment setup.
"""

import os
import logging


lg = logging.getLogger(__name__)


def get_ngrok_token():
    """
    Get ngrok token from environment variable or Colab secrets.

    Returns:
        str: The ngrok auth token

    Raises:
        RuntimeError: If no token is found
    """
    import sys

    # First try environment variable
    token = os.environ.get('NGROK_TOKEN')
    if token:
        return token

    # Then try Colab secrets
    if 'google.colab' in sys.modules:
        try:
            from google.colab import userdata
            token = userdata.get('NGROK_TOKEN')
            if token:
                return token
        except Exception:
            pass

    # No token found, raise helpful error
    raise RuntimeError(
        "NGROK_TOKEN not found. Set it via:\n"
        "  - Environment variable: export NGROK_TOKEN=your_token\n"
        "  - Colab secrets: Add 'NGROK_TOKEN' in the key icon (🔑) sidebar\n\n"
        "Get your token at: https://dashboard.ngrok.com/get-started/your-authtoken"
    )


def create_ngrok_tunnel(port=5006, token=None):
    """
    Create an ngrok tunnel to expose a local port publicly.

    Args:
        port: Local port to tunnel (default: 5006 for Panel)
        token: ngrok auth token. If None, reads from NGROK_TOKEN env var or Colab secrets.

    Returns:
        str: The public ngrok URL

    Raises:
        RuntimeError: If pyngrok is not installed or token is missing
    """
    try:
        from pyngrok import ngrok
    except ImportError:
        raise RuntimeError(
            "pyngrok is not installed. Install it with: pip install pyngrok"
        )

    # Get token
    if token is None:
        token = get_ngrok_token()

    ngrok.set_auth_token(token)

    print(f"🔗 Creating ngrok tunnel for port {port}...")
    public_url = ngrok.connect(port, bind_tls=True)
    ngrok_url = public_url.public_url
    print(f"✓ Public URL: {ngrok_url}")

    return ngrok_url


def close_ngrok_tunnel():
    """
    Close all ngrok tunnels.

    Safe to call even if no tunnel is active.
    """
    try:
        from pyngrok import ngrok
        ngrok.disconnect_all()
        ngrok.kill()
        print("✓ Ngrok tunnel closed")
    except ImportError:
        pass  # pyngrok not installed, nothing to close
    except Exception:
        pass  # Tunnel may not have been active


def check_colab_environment():
    """
    Check if running in Google Colab and validate ngrok token is configured.

    Call this before start_session(ngrok=True) in Colab to get helpful setup instructions
    if the token is missing.

    Raises:
        RuntimeError: If not in Colab or NGROK_TOKEN is not configured
    """
    import sys

    if 'google.colab' not in sys.modules:
        raise RuntimeError("This function is only for Google Colab environment")

    try:
        get_ngrok_token()
        print("✓ NGROK_TOKEN found in Colab Secrets")
    except RuntimeError:
        print("\nIn order to run Vivarium in Colab, you need to use ngrok to enable access to the web interface.")
        print("Here are the steps to set up your ngrok token:")
        print("1. Create an account on ngrok: https://dashboard.ngrok.com/signup")
        print("2. Once you are logged in, go to the 'Auth' section: https://dashboard.ngrok.com/get-started/your-authtoken")
        print("3. Copy your authtoken")
        print("4. In this Colab notebook, click the key icon (🔑) in the left sidebar")
        print("5. Click 'Add a new secret'")
        print("6. Set Name: NGROK_TOKEN")
        print("7. Paste your authtoken as the Value")
        print("8. Toggle on 'Notebook access' for this notebook")
        print("9. Re-run this cell\n")
        raise RuntimeError("NGROK_TOKEN secret not configured")
