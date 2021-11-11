"""Custom nested dataclass for convenient access of nested dataclass attributes."""
# pylint: disable=C0103
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Dict, Union, get_args, get_origin


def nested_dataclass_asdict(datacls: Any) -> Dict:
    """Return a nested dataclass as a dict (convenience function)."""
    if is_dataclass(datacls):
        return asdict(datacls)

    raise RuntimeError(
        "Function should be called with a dataclass instance but was called with {}".format(type(datacls))
    )


def nested_dataclass(*args, **kwargs):
    """
    Nested dataclass annotation.

    Normal dataclasses are difficult to use in a nested way and the types are not correctly inferred.
    Annotating a dataclass with the :code:`nested_dataclass` allows an easy use.
    """

    def wrapper(check_class):
        # passing class to investigate
        check_class = dataclass(check_class, **kwargs)
        _outer_init = check_class.__init__

        class NestedDataclass(check_class):
            @classmethod
            def get_annotations(cls):
                inner_dict = {}
                for class_ in cls.mro():
                    try:
                        inner_dict.update(**class_.__annotations__)
                    except AttributeError:
                        # object, at least, has no __annotations__ attribute.
                        pass
                return inner_dict

            def __init__(self, **kwargs_):
                super().__init__(**kwargs_)
                annotations = self.get_annotations()
                for name, value in kwargs_.items():
                    field_type = annotations.get(name, None)
                    # In case of a Union we must investigate all the options.
                    if get_origin(field_type) is Union and isinstance(value, dict):
                        field_types = get_args(field_type)
                        for field_type in field_types:
                            try:
                                if is_dataclass(field_type):
                                    self._add_to_internal_dict(field_type, name, value)

                            except TypeError:  # One of the illegal union types was called.
                                pass

                    elif is_dataclass(field_type) and isinstance(value, dict):
                        self._add_to_internal_dict(field_type, name, value)

            def _add_to_internal_dict(self, field_type, name: str, value: Dict):
                """Construct an object and add it to the internal dict."""
                obj = field_type(**value)
                self.__dict__[name] = obj

        return NestedDataclass

    return wrapper(args[0]) if args else wrapper
