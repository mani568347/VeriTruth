import re

from django.core.exceptions import ValidationError
from django.utils.translation import gettext as _


SPECIAL_CHARACTERS = "!@#$%^&*()_+-="


class StrictPasswordValidator:
    """Centralized strict password policy for VeriTruth.

    Enforced server-side via AUTH_PASSWORD_VALIDATORS, so it applies to
    registration, password change, and the existing email password-reset form.
    """

    def validate(self, password, user=None):
        errors = []

        if len(password) < 8:
            errors.append(_("Password must be at least 8 characters long."))
        if not re.search(r"[A-Z]", password):
            errors.append(_("Password must contain at least one uppercase letter."))
        if not re.search(r"[a-z]", password):
            errors.append(_("Password must contain at least one lowercase letter."))
        if not re.search(r"[0-9]", password):
            errors.append(_("Password must contain at least one number."))
        if not any(ch in SPECIAL_CHARACTERS for ch in password):
            errors.append(_("Password must contain at least one special character."))

        if errors:
            raise ValidationError(errors)

    def get_help_text(self):
        return _(
            "Your password must be at least 8 characters long and contain at least "
            "one uppercase letter, one lowercase letter, one number, and one special "
            "character (! @ # $ % ^ & * ( ) _ + - =)."
        )
