{{ fullname | escape | underline}}

.. automodule:: {{ fullname }}
   :members:
   :undoc-members:
   :no-inherited-members:
   :exclude-members: declaration_context, definition, database


   {% block attributes %}
   {% if attributes %}
   .. rubric:: Module attributes

   .. autosummary::
      :toctree:
      .. code-block:: python
   {% for item in attributes %}
      {%- if (item not in inherited_members) %}
      {{ item }}
      {%- endif -%}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block classes %}
   {% if classes %}
   .. rubric:: {{ _('Classes') }}

   .. autosummary::
      :toctree:
      :template: custom-class-template.rst

   {% for item in classes %}
      {%- if (item not in inherited_members) %}
      {{ item }}
      {%- endif -%}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block exceptions %}
   {% if exceptions %}
   .. rubric:: {{ _('Exceptions') }}

   .. autosummary::
      :toctree:
      .. code-block:: python
      
   {% for item in exceptions %}
      {%- if (item not in inherited_members) %}
      {{ item }}
      {%- endif -%}
   {%- endfor %}
   {% endif %}
   {% endblock %}


{% block modules %}
{% if modules %}
.. autosummary::
   :toctree:
   :template: custom-module-template.rst
   :recursive:
   

{% for item in modules %}
      {%- if (item not in inherited_members) %}
      {{ item }}
      {%- endif -%}
{%- endfor %}
{% endif %}
{% endblock %}
