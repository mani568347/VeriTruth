from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.password_validation import validate_password


class NewsForm(forms.Form):
    news = forms.CharField(
        label="Enter News Text or URL",
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'rows': 5,
            'placeholder': 'Paste news text or URL...'
        })
    )


class SignupForm(forms.ModelForm):
    password = forms.CharField(widget=forms.PasswordInput)
    confirm_password = forms.CharField(widget=forms.PasswordInput)

    class Meta:
        model = User
        fields = ['username', 'email']

    def clean_username(self):
        username = self.cleaned_data.get('username')
        if not username:
            raise forms.ValidationError("Username is required")
        if User.objects.filter(username=username).exists():
            raise forms.ValidationError("Username already exists")
        return username

    def clean_email(self):
        email = self.cleaned_data.get('email')
        if not email:
            raise forms.ValidationError("Email is required")
        if User.objects.filter(email=email).exists():
            raise forms.ValidationError("Email already used")
        return email

    def clean_password(self):
        password = self.cleaned_data.get('password')
        if not password:
            raise forms.ValidationError("Password is required")
        validate_password(password)  # Django strength rules
        return password

    def clean_confirm_password(self):
        confirm = self.cleaned_data.get('confirm_password')
        if not confirm:
            raise forms.ValidationError("Confirm your password")
        return confirm

    def clean(self):
        cleaned = super().clean()
        p = cleaned.get("password")
        c = cleaned.get("confirm_password")

        if p and c and p != c:
            self.add_error('confirm_password', "Passwords do not match")

        return cleaned

class LoginForm(forms.Form):
    username = forms.CharField()
    password = forms.CharField(widget=forms.PasswordInput)

    def clean(self):
        cleaned_data = super().clean()
        username = cleaned_data.get("username")
        password = cleaned_data.get("password")

        if not username:
            self.add_error('username', "Username is required")

        if not password:
            self.add_error('password', "Password is required")

        return cleaned_data