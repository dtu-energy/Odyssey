<!-- TODO: Fix Botorch examples not formatted properly. -->

On this page, some information and references to the various acquisition functions are provided. 

!!! info "Info"
    If the acquisition function requires the `best_f` parameter, simply set it to 0.0, and Odyssey will find the current best observation by itself.

!!! info "Info"
    The `model`, `posterior_transform` and `maximize` parameters can be ignored.


## **ExpectedImprovement**
Taken from [BoTorch ExpectedImprovement](https://botorch.org/api/acquisition.html#botorch.acquisition.analytic.ExpectedImprovement). 

::: botorch.acquisition.analytic.ExpectedImprovement
    handler: python
    options:
        members:
            - __init__

## **LogExpectedImprovement**
Refer to [BoTorch LogExpectedImprovement](https://botorch.org/api/acquisition.html#botorch.acquisition.analytic.LogExpectedImprovement). 

::: botorch.acquisition.analytic.LogExpectedImprovement
    handler: python
    options:
        members:
            - __init__

## **ProbabilityOfImprovement**
Refer to [BoTorch ProbabilityOfImprovement](https://botorch.org/api/acquisition.html#botorch.acquisition.analytic.ProbabilityOfImprovement). 

::: botorch.acquisition.analytic.ProbabilityOfImprovement
    handler: python
    options:
        members:
            - __init__

## **LogProbabilityOfImprovement**
Refer to [BoTorch LogProbabilityOfImprovement](https://botorch.org/api/acquisition.html#botorch.acquisition.analytic.LogProbabilityOfImprovement). 

::: botorch.acquisition.analytic.LogProbabilityOfImprovement
    handler: python
    options:
        members:
            - __init__

## **UpperConfidenceBound**
Refer to [BoTorch UpperConfidenceBound](https://botorch.org/api/acquisition.html#botorch.acquisition.analytic.UpperConfidenceBound).

::: botorch.acquisition.analytic.UpperConfidenceBound
    handler: python
    options:
        members:
            - __init__