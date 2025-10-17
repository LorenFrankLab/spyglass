
  Spyglass Mixin Class Inheritance Structure

  ===========================================

```
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ BaseMixin (mixins/base.py)                                              │
  │ - Provides: _logger, _test_mode, _spyglass_version, _graph_deps         │
  └─────────────────────────────────────────────────────────────────────────┘
                                      ▲
                                      │
            ┌─────────────────────────┼─────────────────────────┐
            │                         │                         │
            │                         │                         │
  ┌─────────┴──────────┐   ┌──────────┴─────────┐   ┌───────────┴────────┐
  │ CautiousDeleteMixin│   │  PopulateMixin     │   │  RestrictByMixin   │
  │ (mixins/cautious_  │   │  (mixins/populate) │   │  (mixins/restrict_ │
  │  delete.py)        │   │                    │   │   by.py)           │
  └────────────────────┘   └────────────────────┘   └────────────────────┘
                                      │
                           ┌──────────┴──────────┐
                           │                     │
                ┌──────────┴──────────┐   ┌──────┴─────┐
                │  AnalysisMixin      │   │ HelperMixin│
                │  (mixins/analysis)  │   │ (mixins/   │
                │                     │   │  helpers)  │
                └─────────────────────┘   └────────────┘

  ┌─────────────────────────────────────────────────────────────────────────┐
  │ FetchMixin (mixins/fetch.py)                                            │
  │ - Provides: fetch_nwb(), fetch_pynapple(), _nwb_table_tuple             │
  │ - No parent (standalone)                                                │
  └─────────────────────────────────────────────────────────────────────────┘
                                      ▲
                                      │
                           ┌──────────┴──────────┐
                           │  ExportMixin        │
                           │  (mixins/export.py) │
                           │  - Inherits:        │
                           │    FetchMixin       │
                           │  - Adds: _log_fetch,│
                           │    _log_fetch_nwb   │
                           └─────────────────────┘
```

  ═════════════════════════════════════════════════════════════════════════

  MAIN COMPOSITE CLASSES (dj_mixin.py):

```
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ SpyglassMixin                                                           │
  │ - Multiple Inheritance from:                                            │
  │   1. CautiousDeleteMixin                                                │
  │   2. ExportMixin  ──▶  FetchMixin                                       │
  │   3. HelperMixin                                                        │
  │   4. PopulateMixin                                                      │
  │   5. RestrictByMixin                                                    │
  │                                                                         │
  │ - Method Resolution Order (MRO):                                        │
  │   SpyglassMixin → CautiousDeleteMixin → ExportMixin → FetchMixin →      │
  │   HelperMixin → PopulateMixin → RestrictByMixin → BaseMixin → object    │
  └─────────────────────────────────────────────────────────────────────────┘
                                      ▲
                                      │
                    ┌─────────────────┼─────────────────┐
                    │                                   │
           ┌────────┴─────────┐              ┌──────────┴─────────┐
           │ SpyglassMixinPart│              │ SpyglassAnalysis   │
           │ - Inherits:      │              │ - Inherits:        │
           │   SpyglassMixin  │              │   SpyglassMixin    │
           │   dj.Part        │              │   AnalysisMixin    │
           └──────────────────┘              │ - Enforces:        │
                                             │   {prefix}_nwbfile │
                                             │   schema naming    │
                                             └────────────────────┘
```

  ═════════════════════════════════════════════════════════════════════════

  TYPICAL USER TABLE INHERITANCE:

      User Analysis Table (e.g., custom_nwbfile.AnalysisNwbfile)
                      │
                      ▼
      ┌───────────────────────────────────┐
      │  SpyglassAnalysis                 │
      │    │                              │
      │    ├─▶ SpyglassMixin              │
      │    │     └─▶ All 5 mixins         │
      │    │                              │
      │    └─▶ AnalysisMixin              │
      │          └─▶ BaseMixin            │
      │                                   │
      │  + dj.Manual                      │
      └───────────────────────────────────┘

  ═════════════════════════════════════════════════════════════════════════

  KEY RELATIONSHIPS:

  1. BaseMixin is the foundation for most mixins (except FetchMixin)
  2. ExportMixin uniquely inherits from FetchMixin instead of BaseMixin
  3. SpyglassMixin combines 5 mixins to provide full Spyglass functionality
  4. SpyglassAnalysis = SpyglassMixin + AnalysisMixin (both have BaseMixin)
  5. Custom AnalysisNwbfile tables inherit SpyglassAnalysis + dj.Manual
