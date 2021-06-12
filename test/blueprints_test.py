import sys
import classification.blueprints as cb
#import classification.blueprints as blueprints



def test_blueprints(class_to_test):
    class TestBluePrint(class_to_test):
        pass
    return print("The test ran successfully.")



test_blueprints(cb.BluePrint)
