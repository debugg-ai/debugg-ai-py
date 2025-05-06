=============
Top Level API
=============

This is the user facing API of the SDK. It's exposed as ``debugg_ai_sdk``.
With this API you can implement a custom performance monitoring or error reporting solution.

Initializing the SDK
====================

.. autoclass:: debugg_ai_sdk.client.ClientConstructor
   :members:
   :undoc-members:
   :special-members: __init__
   :noindex:

Capturing Data
==============

.. autofunction:: debugg_ai_sdk.api.capture_event
.. autofunction:: debugg_ai_sdk.api.capture_exception
.. autofunction:: debugg_ai_sdk.api.capture_message


Enriching Events
================

.. autofunction:: debugg_ai_sdk.api.add_attachment
.. autofunction:: debugg_ai_sdk.api.add_breadcrumb
.. autofunction:: debugg_ai_sdk.api.set_context
.. autofunction:: debugg_ai_sdk.api.set_extra
.. autofunction:: debugg_ai_sdk.api.set_level
.. autofunction:: debugg_ai_sdk.api.set_tag
.. autofunction:: debugg_ai_sdk.api.set_user


Performance Monitoring
======================

.. autofunction:: debugg_ai_sdk.api.continue_trace
.. autofunction:: debugg_ai_sdk.api.get_current_span
.. autofunction:: debugg_ai_sdk.api.start_span
.. autofunction:: debugg_ai_sdk.api.start_transaction


Distributed Tracing
===================

.. autofunction:: debugg_ai_sdk.api.get_baggage
.. autofunction:: debugg_ai_sdk.api.get_traceparent


Client Management
=================

.. autofunction:: debugg_ai_sdk.api.is_initialized
.. autofunction:: debugg_ai_sdk.api.get_client


Managing Scope (advanced)
=========================

.. autofunction:: debugg_ai_sdk.api.configure_scope
.. autofunction:: debugg_ai_sdk.api.push_scope

.. autofunction:: debugg_ai_sdk.api.new_scope
