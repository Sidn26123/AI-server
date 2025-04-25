from django.urls import path

from .views import (
    DocumentUploadView,
    DocumentListView,
    ConversationCreateView,
    ConversationListView,
    ConversationDetailView,
    ChatView,
)

urlpatterns = [
    path('documents/', DocumentListView.as_view(), name='list_documents'),
    path('documents/upload/', DocumentUploadView.as_view(), name='upload_document'),
    path('conversations/', ConversationListView.as_view(), name='list_conversations'),
    path('conversations/create/', ConversationCreateView.as_view(), name='create_conversation'),
    path('conversations/<int:conversation_id>/', ConversationDetailView.as_view(), name='get_conversation'),
    path('chat/', ChatView.as_view(), name='chat_new'),
    path('chat/<int:conversation_id>/', ChatView.as_view(), name='chat_existing'),
]