# NNCF Architectural Overview

### Introduction
Neural Networks Compression Framework is a set of compression algorithms and tools to implement compression algorithms that is designed to work atop PyTorch.
In essence, all of the compression algorithms present in NNCF do certain manipulations with the data inside the control flow graph of a DNN - be it the process of quantizing the values of an input tensor for a fully connected layer, or setting certain values of a convolutional layer to zero, etc.
A general way to express these manipulations is by using hooks inserted in specific points of the DNN control flow graph.

### NNCFGraph
To abstract away the compression logic from specifics of the backend, NNCF builds an `NNCFGraph` object for each incoming model object to be compressed.
`NNCFGraph` is a wrapper over a regular directed acyclic graph that represents a control flow/execution graph of a DNN. 
Each node corresponds to a call of a backend-specific function ("operator").
It is built both for the original, unmodified model, and for the model with compression algorithms applied (which, in general, may have additional operations when compared to the original model).

### PyTorch-specific

#### NNCFNetwork
During NNCF compression, the incoming original model object is dynamically extended with NNCF-enabling functionality.
This is done by replacing the model object's _class object_ with another class object that lists not only the original class object as its base, but also the `NNCFNetwork` object.
The compressed model object can then be identified as passing the `isinstance(obj, NNCFNetwork)` checks, but also the `isinstance(obj, original_class)` checks.
More specifically, `NNCFNetwork` extends the regular model functionality with an ability to insert hooks into arbitrary places in the model (provided they can be properly addressed in the model control flow graph, see below).
In the model object processed in such a way, the following applies:

1. the `forward` calls are redefined so that the resulting object can function as usual in the training pipeline, but have compression-related functionality called at the right times (before or after native `torch` function calls, or weight accesses in a module).
2. the modules that contain weights are replaced with thin wrappers that allow for better hook handling and trainable compression algorithm parameter storage (see complete list of replaceable modules at `nncf.torch.layers.NNCF_MODULES_DICT`) `Conv2d` modules in the source model, for instance, will be replaced with `NNCFConv2d` modules upon wrapping the model with `NNCFNetwork`. This does not impact the module's original weights and parameters and their saving or loading from a PyTorch checkpoint.
3. additional trainable modules and parameters specific to the applied compression algorithms are invisibly stored along with the regular model parameters, so that when saving an instance of `NNCFNetwork` via the usual `torch.save` calls, the trainable parameters of the compression algorithm are saved into the same state dict as the rest of the model parameters.

The additional attributes and methods that appear in the original model object are separated from the original attribute/method names - the accesses to the NNCF-specific attributes and methods are done via an intermediate `nncf` property:
```python3
assert isinstance(model, NNCFNetwork)
model.original_method_call()
model.nncf.nncf_specific_method()
```
This allows to avoid name collisions between NNCF-specific attributes and original model attributes. 

`model.nncf` returns a `nncf.torch.nncf_network.NNCFNetworkInterface` object - the class contains all of the methods and attributes that could be called on the compressed model object to invoke NNCF-specific functionality.

During compression algorithm application, the `NNCFNetwork` serves internally as a receptacle for compression algorithm-related adjustments to the control flow graph of the model.


#### Model control flow graph tracing
Unlike other frameworks such as TensorFlow, PyTorch does not have an easily accessible graph representation of a model, and thus no way to identify specific points in the control flow graph.
For this reason NNCF performs tracing of the PyTorch operators, implemented via wrapping the corresponding function and module calls.
Through this process of tracing, NNCF builds an internal representation of the model graph, which is then supplied as the point of reference for specification and insertion of hooks at proper places in the network.

The `NNCFGraph` is built by executing a model's `forward` call once with mock inputs within a `TracingContext`.
A `forward` call of the model eventually triggers calls of the wrapped PyTorch operators (Python functions) within the model; once each operator is called, the `TracingContext` assigns the current call to a node in the `NNCFGraph`.
It does so by keeping track of

a) the location of the current operator call inside the encompassing model's module hierarchy, i.e. a scope (represented by a `Scope` class objects in the NNCF),

b) the order of calls to the same operator within the same scope (to distinguish multiple `torch.nn.functional.ReLU` calls in a "for" loop, for example),

c) the shape of the input tensors to the current operator, and

d) the IDs of the nodes that produced each current operator's input as their output.


This information is stored as an `OperationExecutionContext` of the operator. If an operator call does not match to the nodes already present in the internal graph representation based on its `OperationExecutionContext`, a new node is added to the graph.
This process occurs dynamically during each `forward` call of an `NNCFNetwork`. 
If the control flow is data-dependent, a whole new subgraph of the model will be built for each branching in the model definition. 
The graph building mechanism can cope with some branching, but it is advisable to disable NNCF tracing for the parts of the model that exhibit branching (such as the "detection output" layers of object detection networks) by using a `no_nncf_trace()` context. 
It is possible to wrap third party functionality with `no_nncf_trace()` context so that this source code does not need to be changed. 
This can be done by patching, please refer to this [example](../examples/post_training_quantization/torch/ssd300_vgg16/README.md).



#### Operation scope and addressing
A unique identifier of a node in the `NNCFGraph` - i.e. an operation in the DNN control flow graph - is the `OperationExecutionContext`.
However, in most cases the input-agnostic part of `OperationExecutionContext` is enough to identify an operation in the model control flow graph for purposes of inserting compression-related hooks into the model.
This `InputAgnosticOperationExecutionContext` is built using a) and b) from the information list gathered to build a regular `OperationExecutionContext`. Its string representation is a concatenation of a `Scope` string representation, the name of the operator (Python function), underscore `_`, and the order of the operator call in the same `Scope`. In turn, the string representation of a `Scope` is a sequence of "__module_class_name__[__module_field_name__]/" substrings, where each such substring corresponds to a __module_class_name__ type of `torch.nn.Module` being called as a __module_field_name__ member field of its parent module, and slashes `/` separate the adjacent levels of the module call hierarchy.

As an example, consider a simple PyTorch module:

```python
class SimpleModule(torch.nn.Module):
	def __init__():
		super().__init__()
		self.submodule1 = torch.nn.Conv2d(...) # params omitted
		self.submodule2 = torch.nn.Sequential([torch.nn.BatchNorm2d(...), torch.nn.ReLU(...)])
	def forward(x_in):
		x = self.submodule1(x_in)
		x = self.submodule2(x)
		x += torch.ones_like(x)
		x += torch.ones_like(x)
		x = torch.nn.functional.relu(x)
		return x
```

Each `torch.nn.Conv2d` module call internally calls a `conv2d` operator, which will then be added to an `NNCFGraph` during tracing. Therefore, the two convolution operations in the model's control flow graph will have the following `InputAgnosticOperationExecutionContext` string representations: `SimpleModule/Conv2d[submodule1]/conv2d_0` and `SimpleModule/Conv2d[submodule2]/conv2d_0`.

The `torch.nn.Sequential` layer uses indices for addressing into the internal object, so `torch.nn.BatchNorm` will be represented by `SimpleModule/Sequential[submodule2]/BatchNorm2d[0]/batch_norm_0`, and `torch.nn.ReLU` - by `SimpleModule/Sequential[submodule2]/ReLU[1]/relu_0`.

The two consecutive addition operations will be represented by `SimpleModule/__iadd___0` and `SimpleModule/__iadd___1` respectively, and the final ReLU call, which happens outside the ReLU module, will be represented by `SimpleModule/relu_0`.

These string definitions are referred to as "scopes" in the NNCF configuration files (as in `"ignored_scopes"` or `"target_scopes"`), and help specify exact operations for inclusion into or exclusion from compression or for separate compression parameter specification.



#### Compression algorithm API and interaction with NNCFNetwork
A compression algorithm is a modification of a regular model control flow according to some trainable or non-trainable parameters. Modification of the control flow is done via a hook, and the trainable parameters are stored inside special NNCF modules. Each compression algorithm therefore consists of taking an unmodified model, analyzing it and then determining a set of modifications necessary for modifying the model's execution so that it now takes specific compression into account.

`NNCFNetwork` defines a common interface for compression algorithms to specify the location for hook insertion (based on a `InputAgnosticOperationExecutionContext` of an operation) and the hook itself. It also allows algorithms to register external modules within itself so that the trainable parameters of the compression algorithm could be saved as a checkpoint along with the model while also being indistinguishable from any other trainable parameter of the original model from the training pipeline optimizer's standpoint.

The application of a compression algorithm to a model and subsequent control of the compression algorithm parameters is implemented via a `Builder`/`Controller` architecture. An implementation of each algorithm should define both a `CompressionAlgorithmBuilder` class, which takes an unmodified `NNCFNetwork` model view as input to its `.apply_to(nncf_network)` function. Within the `CompressionAlgorithmBuilder.apply_to` usually produces a list of modifications (`InsertionCommand`s) which are required to apply the algorithm to a model, and then calls the input `NNCFNetwork`'s method `register_insertion_command` to register these changes. Finally, the `CompressionAlgorithmBuilder` should register itself as being applied to the model by calling `NNCFNetwork.register_algorithm`.

A modification consists of inserting a specified hook at a specified location. A compression algorithm can, in theory, insert hooks either before a node ("operation pre-hooks") or after a node ("operation post-hooks") in the `NNCFGraph`. This is mostly used to quantize the activations of the model.
Weight quantization is currently done separately by registering a hook to be executed prior to a module call ("module pre-op") or after a module call ("module post-op"). In case the hook needs to call a compression algorithm-specific module with trainable parameters, the `Builder` should register the corresponding module within the `NNCFNetwork`.

The algorithm does not apply the control flow graph changes (hook registration) itself, as it may potentially ruin the subsequent application of other compression algorithms; it only registers the changes and associated modules.

Multiple algorithms can be applied consequently to the same `NNCFNetwork` instance. Hooks that belong to different compression algorithms may be inserted into the same locations. These will be executed according to priority specified in an `InsertionCommand`, so to ensure correct interoperability of different compression algorithms when applied simultaneously, care should be taken while specifying correct hook priority at each algorithm's `apply_to` stage.

Once all algorithms are applied to the model, the compression changes are committed into the `NNCFNetwork` by calling its `commit_compression_changes()`. It registers all the necessary hooks into the model properly (so that they are called during next `forward` function call), and produces a `CompressionAlgorithmController` object - a handle to the applied compression algorithm(s). The controller object is built by calling a `build_controller` function from each `CompressionAlgorithmBuilder` that acted on the `NNCFNetwork`; if multiple compression algorithms were applied to the model, then each `CompressionAlgorithmController` is composed and a single `CompositeCompressionController` object is returned.

A `CompressionAlgorithmController` is then used to control or modify aspects of compression during training, to gather statistics related to compression, or to provide additional loss for proper training of the trainable compression parameters. To this purpose it contains a `CompressionScheduler` and `CompressionLoss` instances, which can then be used as desired during the training pipeline. For instance, a `CompressionScheduler` may be implemented so that it enables quantization for activations only upon a certain training epoch, and a `CompressionLoss` may be implemented so that it facilitates soft filter pruning.

> **NOTE**: In general, the compression method may not have its own scheduler and loss, and the default implementations are used instead.
