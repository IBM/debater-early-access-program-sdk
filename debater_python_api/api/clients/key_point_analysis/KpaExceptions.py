
class KpaIllegalInputException(Exception):
    '''
    This exception is thrown when the user passes an illegal input.
    '''
    pass

class KpaNoPrivilegesException(Exception):
    '''
    This exception is thrown when the user don't have privilege for the requested operation (admin operations).
    '''
    pass