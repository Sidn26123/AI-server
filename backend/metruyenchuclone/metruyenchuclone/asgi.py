"""
ASGI config for metruyenchuclone project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.0/howto/deployment/asgi/
"""

import os

from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from django.urls import path

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'metruyenchuclone.settings')

# application = get_asgi_application()

application = ProtocolTypeRouter({
    "http": get_asgi_application(),
    "websocket": 
        URLRouter(
            [
                # path('ws/chat/<str:room_name>/', ChatConsumer.as_asgi()),
            ]
        )
})
