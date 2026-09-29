from django.urls import path
from . import views
from django.contrib.auth import views as auth_views


urlpatterns = [
    path('', views.home, name='home'),
    path('predict/', views.predict),
    path('result/', views.result),
    path('history/', views.history, name='history'),
    path('history/<int:id>/', views.history_detail),
    path('ask-ai/', views.ask_ai),
    path('feedback/submit/', views.submit_feedback, name='feedback_submit'),
    path('admin-feedback/update/', views.admin_feedback_update, name='admin_feedback_update'),
    path('signup/', views.signup_view, name='signup'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('about/', views.about, name='about'),
    path('verify/<uidb64>/<token>/', views.verify_email, name='verify_email'),
    path('terms/', views.terms, name='terms'),
    path('privacy/', views.privacy, name='privacy'),
    path('support/', views.support, name='support'),
    path('faq/', views.faq, name='faq'),
    path('disclaimer/', views.disclaimer, name='disclaimer'),
   
    path(
    "api/history/",
    views.api_history
),

    path(
        'verify/<uidb64>/<token>/',
        views.verify_email,
        name='verify_email'
    ),

    # PASSWORD RESET

path(
    'password-reset/',
    views.CustomPasswordResetView.as_view(),
    name='password_reset'
),

 path(
    'reset/done/',
    views.CustomPasswordResetCompleteView.as_view(),
    name='password_reset_complete'
),  

    path(
        'reset/<uidb64>/<token>/',
        auth_views.PasswordResetConfirmView.as_view(
            template_name='password_reset_confirm.html'
        ),
        name='password_reset_confirm'
    ),

    path(
        'reset/done/',
        auth_views.PasswordResetCompleteView.as_view(
            template_name='password_reset_complete.html'
        ),
        name='password_reset_complete'
    ),

    path('admin-dashboard/', views.admin_dashboard, name='admin_dashboard'),
    path('admin-login/', views.admin_login_view, name='admin_login'),
    path('api/chat-assistant/', views.chat_assistant_view, name='chat_assistant'),
]



