# Jupyter configuration to allow iframe embedding
# This allows the notebook to be embedded in Panel UI

c = get_config()  # noqa

# Allow all origins (for local development)
c.NotebookApp.allow_origin = '*'

# Disable XSRF checks for iframe embedding (local development only)
c.NotebookApp.disable_check_xsrf = True

# Allow iframe embedding by setting appropriate headers
c.NotebookApp.tornado_settings = {
    'headers': {
        'Content-Security-Policy': "frame-ancestors 'self' http://localhost:* http://127.0.0.1:*"
    }
}

# Optional: Disable token for easier local development
# c.NotebookApp.token = ''
# c.NotebookApp.password = ''
