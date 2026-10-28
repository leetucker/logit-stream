# Results

The result classes subclass hdfe-stream's `RegressionResult`, so the
reporting methods (`summary`, `tidy`, `confint`, `wald_test`, `with_vcov`,
`summary_dict`, `summary_json`, `to_pyfixest`) are hdfe-stream's own and
behave identically. The binary logit returns hdfe-stream's `HDFEResult`
([its reference](https://leetucker.github.io/hdfe-stream/reference/results.html)).

```{eval-rst}
.. currentmodule:: logit_stream

.. autoclass:: MultinomialLogitResult
   :no-members:

   .. automethod:: summary
   .. automethod:: summary_text
   .. automethod:: summary_dict
   .. automethod:: summary_json
   .. automethod:: tidy
   .. automethod:: coef
   .. automethod:: coef_matrix
   .. automethod:: confint
   .. automethod:: wald_test
   .. automethod:: with_vcov
   .. automethod:: predict
   .. automethod:: resid
   .. automethod:: fixef
   .. automethod:: ranef
   .. automethod:: varcorr
   .. automethod:: sample
   .. automethod:: to_pyfixest

.. autoclass:: OrderedLogitResult
   :no-members:

   .. autoproperty:: cutpoints
   .. automethod:: predict
   .. automethod:: resid
   .. automethod:: fixef
   .. automethod:: ranef
   .. automethod:: varcorr
   .. automethod:: sample

.. autoclass:: OrderedProbitResult
   :no-members:

.. autoclass:: ConditionalLogitResult
   :no-members:

   .. automethod:: predict
   .. automethod:: resid
   .. automethod:: sample

.. autoclass:: ConditionalFELogitResult
   :no-members:

   .. automethod:: predict

.. autoclass:: MixedLogitResult
   :no-members:

.. autoclass:: NestedLogitResult
   :no-members:

.. autoclass:: LogitResult
   :no-members:

.. autoclass:: LogitMulti
   :no-members:

   .. automethod:: fetch_model
   .. automethod:: tidy
   .. automethod:: summary
   .. automethod:: summary_dict
   .. automethod:: summary_json
```
