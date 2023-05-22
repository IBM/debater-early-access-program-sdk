
class KpsIllegalInputException(Exception):
    '''
    This exception is thrown when the user passes an illegal input.
    '''
    pass


class KpsNoPrivilegesException(Exception):
    '''
    This exception is thrown when the user don't have privilege for the requested operation (admin operations).
    '''
    pass


class KpsInvalidOperationException(Exception):
    '''
    This exception is thrown when the user tries to perform an invalid kps operation.
    '''
    pass