### Q & A

1. python NameError: name 'file' is not defined

 A: file() is no supported in python3, use open() instead

2. pydot.InvocationException: GraphViz's executables not found

 A: You need to install from Graphviz and then just add the path of folder where you installed Graphviz and its bin directory to system environments **path**. Note: If multiple users is opened in your PC, you'd better install graphviz for all users, as windows os set the owner user as default when adding to path.

3. TypeError: sequence item 0: expected str instance, bytes found

 A: You can use `b''.join()` to join a byte sequence but based on what you wan you can also use map(str,seq) to convert your byte sequence items to string.