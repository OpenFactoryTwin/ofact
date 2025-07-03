# State Model Consistency

A common problem in industry projects is the inconsistency of data, provided from several systems/ data sources.
Especially for the simulation, the state model must be consistent. 
If this is not the case, the simulation crashes.

## Inconsistencies

Coming from the problem, the solution would be to ensure the consistency of the data in the state model, 
before the simulation.
Therefore, the consistency should be ensured on different levels:
- Each object in the state model should be **completely filled**, considering the obligatory fields/attributes.
- The process chains should be **logically consistent**
    - **No Data Gaps** (e.g., after process A, process B should continue and not C, 
        since process B would be missing in the process chain)
    - **No logically impossible chain sequences** 
        (e.g., after process B, process C or D should continue and not process A, if this is defined in the logic)

However, it should be always tried to ensure the data consistency as early as possible, 
to avoid aggregated failures as well as correct the error quickly after detection.

## Solution Design

To handle inconsistencies, two steps are gone through.
Firstly, the data is validated (**I. validation**). In this phase, all possible errors are detected.
Secondly, the detected inconsistencies are resolved to ensure the eventual consistency (**II. ensurance**).
