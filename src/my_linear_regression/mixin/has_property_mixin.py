#!/usr/bin/env python3.8.5
# Copyright 2020, Rose Software Ltd, All rights reserved.


class HasPropertyMixin():
    """
    This mixin can be used to extend objects with the ability to
    inspect whether a given property exists for that object.
    """

    def has_object(
            self,
            name: str) -> bool:
        """
        Checks whether the class that contains the mixin has a object
        with the name. \n
        Returns:
            bool
        Doctest:
            >>> hpm = HasPropertyMixin()
            >>> assert hpm.has_object('has_object') == True
        """
        return len([
                    obj
                    for obj in dir(self)
                    if obj == name])

        
if __name__ == "__main__":
    import doctest
    doctest.testmod()
