## **Base Navigator**

::: odyssey.navigators.Navigator
    handler: python
    options:
        members:
            - __init__
            - trajectory
            - probe
            - relay
            - upgrade
        line_length: 5


## **Single Objective Navigators**

::: odyssey.navigators.SingleGP_Navigator
    handler: python
    options:
        members:
            - __init__

## **Multi Objective Navigators**
::: odyssey.navigators.moo.qNParEGO_Navigator
    handler: python
    options:
        members:
            - __init__

::: odyssey.navigators.moo.Gryffin_Navigator
    handler: python
    options:
        members:
            - __init__


::: odyssey.navigators.moo.Dragonfly_Navigator
    handler: python
    options:
        members:
            - __init__


::: odyssey.navigators.moo.Falcon_Navigator
    handler: python
    options:
        members:
            - __init__

## **Sampler Navigators**
::: odyssey.navigators.sampler_navigators.Sobol_Navigator
    handler: python
    options:
        members:
            - __init__

::: odyssey.navigators.sampler_navigators.Grid_Navigator
    handler: python
    options:    
        members:
            - __init__

## **DataLoader**
::: odyssey.navigators.dataloader.DataLoader
    handler: python
    options:
        members:
            - __init__



                
