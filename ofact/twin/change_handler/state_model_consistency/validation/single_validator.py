

def validate_object(single_object):
    completely_filled, not_completely_filled_attributes = single_object.completely_filled()

    return completely_filled, not_completely_filled_attributes
