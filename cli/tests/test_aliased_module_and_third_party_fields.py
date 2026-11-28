"""Regression tests for two field-detection gaps surfaced by targeted fuzz:

1. **Aliased models module.** ``from django.db import models as m`` followed
   by ``class X(m.Model): x = m.CharField(...)`` used to lose every field
   because ``FIELD_RE`` hardcoded the ``models\\.`` prefix and
   ``BARE_FIELD_RE`` had no prefix allowance at all.

2. **Third-party field packages.** Fields declared via a namespaced
   third-party import - ``x = jsonfield.JSONField(default=dict)``,
   ``x = timezonefield.TimeZoneField()`` - also fell through the same
   two-way gap.

Fixed by adding an optional single-identifier prefix on the RHS in
``BARE_FIELD_RE``. Safe against false positives because the type name
is still restricted to Django's known field whitelist
(``BARE_FIELD_TYPES``) - random ``foo.CharField(...)`` calls in
non-Django code won't leak in unless the trailing name is a real
Django field type.
"""

from __future__ import annotations

import unittest

from orm_lens.parser import parse_models_file


class AliasedModelsModuleTest(unittest.TestCase):
    def test_all_fields_captured_via_alias(self) -> None:
        src = (
            "from django.db import models as m\n\n"
            "class Order(m.Model):\n"
            "    status = m.CharField(max_length=20)\n"
            "    total = m.DecimalField(max_digits=10, decimal_places=2)\n"
            "    quantity = m.IntegerField(default=1)\n"
        )
        models = parse_models_file("shop/models.py", src)
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0].name, "Order")
        self.assertEqual(
            [f.name for f in models[0].fields],
            ["status", "total", "quantity"],
        )
        self.assertEqual(models[0].fields[0].type, "CharField")

    def test_mixed_alias_and_plain_and_bare(self) -> None:
        # One model, three import styles interleaved - every field must
        # be captured regardless of prefix.
        src = (
            "from django.db import models as m\n"
            "from django.db.models import IntegerField\n\n"
            "class Row(m.Model):\n"
            "    aliased = m.CharField(max_length=10)\n"
            "    plain = models.TextField()\n"
            "    bare = IntegerField(default=0)\n"
        )
        models = parse_models_file("t/models.py", src)
        self.assertEqual(
            [f.name for f in models[0].fields],
            ["aliased", "plain", "bare"],
        )


class ThirdPartyFieldPackageTest(unittest.TestCase):
    def test_jsonfield_style_package(self) -> None:
        # `django-jsonfield`-style: `x = jsonfield.JSONField(default=dict)`.
        # JSONField is in the standard whitelist, so this must match even
        # under a non-`models.` prefix.
        src = (
            "import jsonfield\n\n"
            "class Config(models.Model):\n"
            "    data = jsonfield.JSONField(default=dict)\n"
        )
        models = parse_models_file("t/models.py", src)
        self.assertEqual([f.name for f in models[0].fields], ["data"])
        self.assertEqual(models[0].fields[0].type, "JSONField")

    def test_plain_bare_import_still_works(self) -> None:
        # Backward-compat: the pre-fix bare form (no prefix at all) must
        # still match - the prefix is truly optional, not required.
        src = (
            "from django.db.models import CharField, IntegerField\n\n"
            "class R(models.Model):\n"
            "    name = CharField(max_length=50)\n"
            "    count = IntegerField(default=0)\n"
        )
        models = parse_models_file("t/models.py", src)
        self.assertEqual(
            [f.name for f in models[0].fields], ["name", "count"]
        )


class TaggableManagerTest(unittest.TestCase):
    def test_default_relation_metadata(self) -> None:
        src = (
            "from django.db import models\n"
            "from taggit.managers import TaggableManager\n\n"
            "class Post(models.Model):\n"
            "    title = models.CharField(max_length=200)\n"
            "    tags = TaggableManager()\n"
        )
        models = parse_models_file("blog/models.py", src)
        tags = models[0].fields[1]
        self.assertEqual(tags.type, "TaggableManager")
        self.assertTrue(tags.is_relation)
        self.assertEqual(tags.relation_kind, "ManyToManyField")
        self.assertEqual(tags.related_model, "taggit.Tag")
        self.assertEqual(tags.through_model, "taggit.TaggedItem")

    def test_explicit_through_overrides_default(self) -> None:
        src = (
            "from django.db import models\n"
            "from taggit.managers import TaggableManager\n\n"
            "class Post(models.Model):\n"
            "    tags = TaggableManager(through=CustomTaggedItem)\n"
        )
        models = parse_models_file("blog/models.py", src)
        tags = models[0].fields[0]
        self.assertEqual(tags.related_model, "taggit.Tag")
        self.assertEqual(tags.through_model, "CustomTaggedItem")


class ThirdPartyFieldClassesTest(unittest.TestCase):
    """Issue #377: field classes from site-packages, imported bare."""

    def test_third_party_field_classes_are_fields(self) -> None:
        src = (
            "from django.db import models\n"
            "from i18nfield.fields import I18nCharField, I18nTextField\n"
            "from phonenumber_field.modelfields import PhoneNumberField\n"
            "from django_countries.fields import CountryField\n"
            "from pgvector.django import HalfVectorField, SparseVectorField, VectorField\n\n"
            "class Event(models.Model):\n"
            "    name = I18nCharField(max_length=200)\n"
            "    about = I18nTextField(blank=True)\n"
            "    phone = PhoneNumberField(blank=True)\n"
            "    country = CountryField()\n"
            "    embedding = VectorField(dimensions=3)\n"
            "    half = HalfVectorField(dimensions=3)\n"
            "    sparse = SparseVectorField(dimensions=3)\n"
        )
        fields = parse_models_file("events/models.py", src)[0].fields
        self.assertEqual(
            [(f.name, f.type, f.is_relation) for f in fields],
            [
                ("name", "I18nCharField", False),
                ("about", "I18nTextField", False),
                ("phone", "PhoneNumberField", False),
                ("country", "CountryField", False),
                ("embedding", "VectorField", False),
                ("half", "HalfVectorField", False),
                ("sparse", "SparseVectorField", False),
            ],
        )

    def test_sorted_many_to_many_is_a_many_to_many_relation(self) -> None:
        src = (
            "from django.db import models\n"
            "from sortedm2m.fields import SortedManyToManyField\n\n"
            "class Template(models.Model):\n"
            "    name = models.CharField(max_length=200)\n\n"
            "class Config(models.Model):\n"
            "    templates = SortedManyToManyField(Template, blank=True, related_name='configs')\n"
        )
        config = parse_models_file("shop/models.py", src)[1]
        self.assertEqual([f.name for f in config.fields], ["templates"])
        templates = config.fields[0]
        self.assertTrue(templates.is_relation)
        self.assertEqual(templates.relation_kind, "ManyToManyField")
        self.assertEqual(templates.related_model, "Template")
        self.assertEqual(templates.related_name, "configs")
        self.assertIsNone(templates.on_delete)


if __name__ == "__main__":
    unittest.main()
