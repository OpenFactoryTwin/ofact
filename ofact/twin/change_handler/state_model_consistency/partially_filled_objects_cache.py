"""Used to cache partially filled objects because, e.g., not all required information is available until now."""


class PartiallyFilledObjects:

    def __init__(self):
        """
        Used to track only partially filled objects of the digital twin.
        They should be completely filled in the following updates/ changes.

        Parameters
        ----------
        attribute objects_with_not_completed_attributes: a dict that maps a list of not completed attributes
        to the related object
        """
        self.objects_with_not_completed_attributes: dict[object, list[str]] = {}

    def add_partially_filled_object(self, object_, not_completely_filled_attributes=None):
        if not_completely_filled_attributes is None:
            completely_filled, not_completely_filled_attributes = object_.completely_filled()
        self.objects_with_not_completed_attributes[object_] = not_completely_filled_attributes

    def update_partially_filled_object(self, object_, completely_filled=None, not_completely_filled_attributes=None):
        if object_ not in self.objects_with_not_completed_attributes:
            return

        if completely_filled is None or not_completely_filled_attributes is None:
            completely_filled, not_completely_filled_attributes = object_.completely_filled()

        if completely_filled:
            self._remove_completely_filled_object(object_)
        else:
            self.objects_with_not_completed_attributes[object_] = not_completely_filled_attributes

    def _remove_completely_filled_object(self, object_):
        del self.objects_with_not_completed_attributes[object_]

    def remove_partially_filled_object(self, object_):
        del self.objects_with_not_completed_attributes[object_]
