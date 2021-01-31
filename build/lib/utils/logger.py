"Write to a log file everything passed to print()"
import sys
import warnings
import numpy as np
class Logger():
    """Write to a log file everything passed to print().

    Parameters
    ----------
    logfile : str
        path to the log file.

    Examples
    -------
    Define the logger
    >>> logger = Logger(logfile)

    Start logging sys.stdout
    >>> logger.open()

    End logging sys.stdout
    >>> logger.close()

    Attributes
    ----------
    terminal :
        local storage of original sys.stdout.
    log :
        file handler for logfile.

    """
    #****   Constant declaration    ****#
    TitleLength=40

    #****   Core methods    ****#
    def __init__(self, logfile: str):
        self.terminal=None
        self.log = None
        self.logfile=logfile
    def __getattr__(self, attr):
            return getattr(self.terminal, attr)
    def write(self, message):
        """Overwrite write method to enable writing in both sys.stdout and log file."""
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
    def close(self):
        """Close log file and restore sys.stdout"""
        self.log.close()
        sys.stdout = self.terminal
    def open(self):
        """Open log file and save sys.stdout"""
        self.terminal = sys.stdout
        self.log = open(self.logfile, "a+")
        sys.stdout = self

    #****   Additionnal methods related to printing    ****#
    def printTitle(title, titleLen = None, newLine=True):
        """Print out title header"""
        if titleLen == None: titleLen = Logger.TitleLength
        if len(title)>titleLen: warnings.warn("In makeTitle, title length bigger than %s"%(titleLen))
        print("="*np.max([0,np.int((titleLen-len(title))/2)+(titleLen-len(title))%2])+" "+title+" "+"="*np.max([0,np.int((titleLen-len(title))/2)]))
        if newLine: print('\n')
    def printDic(dic,skipKeys=None):
        """Print dictionary nicely, skipping the 'skipKeys' keys"""
        if skipKeys==None: skipKeys=['']
        for k in dic:
            if k in skipKeys: continue
            print("\t",k,"-"*(1+len(max(dic,key=len))-len(k)),">",dic[k])
        print('\n')
