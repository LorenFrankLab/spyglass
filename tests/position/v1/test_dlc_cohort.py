import pytest


def test_cohort_pop(sgp, cohort_key):
    cohort_tbl = sgp.v1.DLCSmoothInterpCohort()
    assert len(cohort_tbl), "Cohort table failed to populate"


@pytest.fixture(scope="session")
def cohort_tbls(sgp, cohort_key):
    _ = cohort_key
    select_tbl = sgp.v1.DLCSmoothInterpCohortSelection()
    cohort_tbl = sgp.v1.DLCSmoothInterpCohort()

    yield select_tbl, cohort_tbl


def test_cohort_null_params(cohort_tbls):
    select_tbl, cohort_tbl = cohort_tbls

    select_key = select_tbl.fetch(limit=1, as_dict=True)[0]
    select_key.update(
        dict(
            dlc_si_cohort_selection_name="test no bodyparts",
            bodyparts_params_dict=dict(),
        )
    )
    select_tbl.insert1(select_key, skip_duplicates=True)
    cohort_tbl.populate(select_key)
    part_tbl = cohort_tbl.BodyPart & select_key

    assert len(part_tbl) == 0, "Cohort table populated w/empty bodyparts params"


def test_cohort_error(cohort_tbls):
    select_tbl, cohort_tbl = cohort_tbls
    select_key = select_tbl.fetch(limit=1, as_dict=True)[0]
    select_pk = dict(dlc_si_cohort_selection_name="test bad bodyparts")
    select_key.update(
        dict(select_pk, bodyparts_params_dict=dict(bad_bp="bad_bp"))
    )
    if select_tbl & select_pk:
        select_tbl.delete(safemode=False)
    select_tbl.insert1(select_key, skip_duplicates=True)

    with pytest.raises(ValueError):
        cohort_tbl.populate(select_key)
