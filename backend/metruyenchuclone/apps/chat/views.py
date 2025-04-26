from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.shortcuts import get_object_or_404
from .models import Document, Conversation, Message
from .utils import ollama_chat, get_embeddings_cache

class DocumentUploadView(APIView):
    def post(self, request):
        try:
            data = request.data
            document = Document.objects.create(
                title=data.get('title', 'Untitled Document'),
                content=data.get('content', '')
            )
            get_embeddings_cache()
            return Response({
                'id': document.id,
                'title': document.title,
                'message': 'Document uploaded successfully'
            }, status=status.HTTP_201_CREATED)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)

class DocumentListView(APIView):
    def get(self, request):
        documents = Document.objects.all()
        return Response([{
            'id': doc.id,
            'title': doc.title,
            'created_at': doc.created_at
        } for doc in documents])

class ConversationCreateView(APIView):
    def post(self, request):
        try:
            title = request.data.get('title', 'New Conversation')
            conversation = Conversation.objects.create(title=title)
            system_message = request.data.get('system_message')
            if system_message:
                Message.objects.create(
                    conversation=conversation,
                    role='system',
                    content=system_message
                )
            return Response({
                'id': conversation.id,
                'title': conversation.title,
                'created_at': conversation.created_at
            }, status=status.HTTP_201_CREATED)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)

class ConversationListView(APIView):
    def get(self, request):
        conversations = Conversation.objects.all()
        return Response([{
            'id': conv.id,
            'title': conv.title,
            'created_at': conv.created_at
        } for conv in conversations])

class ConversationDetailView(APIView):
    def get(self, request, conversation_id):
        try:
            conversation = get_object_or_404(Conversation, id=conversation_id)
            messages = conversation.messages.all()
            return Response({
                'id': conversation.id,
                'title': conversation.title,
                'created_at': conversation.created_at,
                'messages': [
                    {
                        'id': msg.id,
                        'role': msg.role,
                        'content': msg.content,
                        'timestamp': msg.timestamp
                    } for msg in messages
                ]
            })
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)

class ChatView(APIView):
    def post(self, request, conversation_id=None):
        try:
            if conversation_id:
                conversation = get_object_or_404(Conversation, id=conversation_id)
            else:
                conversation = Conversation.objects.create(title="New Conversation")

            user_input = request.data.get('message', '')
            system_messages = conversation.messages.filter(role='system')
            if system_messages.exists():
                system_message = system_messages.first().content
            else:
                system_message = "You are a helpful assistant that is an expert at extracting the most useful information from a given text. Also bring in extra relevant information to the user query from outside the given context."
                Message.objects.create(
                    conversation=conversation,
                    role='system',
                    content=system_message
                )

            history = [
                {'role': msg.role, 'content': msg.content}
                for msg in conversation.messages.exclude(role='system')
            ]

            ollama_model = request.data.get('model', 'llama3.2:1b')
            chat_result = ollama_chat(
                user_input=user_input,
                system_message=system_message,
                conversation_history=history.copy(),
                ollama_model=ollama_model
            )

            Message.objects.create(conversation=conversation, role='user', content=user_input)
            Message.objects.create(conversation=conversation, role='assistant', content=chat_result['response'])

            return Response({
                'conversation_id': conversation.id,
                'original_query': chat_result['original_query'],
                'rewritten_query': chat_result['rewritten_query'],
                'relevant_context': chat_result['relevant_context'],
                'response': chat_result['response']
            })
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
