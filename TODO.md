# spikesorting v2 TODO before merge
1. Make sure we are using all the spyglass machinery
2. no phase/plan etc in comments, function names, test names.
3. Ensure all integration tests cover the common cases (particuarly the curation scenarios)
4. Make sure we are using `delete` and not `quick_delete`, etc unless we have a good reason. Don't want to bypass cautious delete machinery.
5. Make sure all the docs are up to date and plan docs are gone.
6. if there's a chance to refactor logic where there is complicated code trying to get around a problem, but there's a simpler, cleaner, clearer, more maintainable and/or more efficient way.
7. Evaluate the names of everything, and make sure they are clear, consistent, and follow the naming conventions.
8. Make sure we have documented what is the same as v1 and what is different
9. Look for hardening against impossible cases or overly compplicated solutions.
10. Are things implemented in the same way? Are we being consistent? Are we using the same patterns and approaches for similar problems?