/* Spyglass review controls for the pinned FigPack context API.
 * Scientific commits use the connected Python review facade.
 * Hosted figures retain FigPack's authenticated annotation-saving toolbar.
 */
const reviewSessions = new WeakMap();
window.figpack_p1.registerFPViewComponent({
  name: "spyglass.ReviewControls",
  render: async ({container, zarrGroup, contexts}) => {
    const annotations = contexts.figureAnnotations;
    const selection = contexts.unitSelection;
    const reducer = contexts.sortingCuration;
    const local = window.location.hostname === "localhost" &&
      !new URLSearchParams(window.location.search).has("figure");
    const seed = zarrGroup.attrs.curation;
    const palette = zarrGroup.attrs.label_options;
    let session = reviewSessions.get(annotations);
    const firstMount = !session;
    if (!session) {
      session = {saved: null, saving: false, error: "", loaded: !local, render: () => {},
        connected: false, reviewId: null, operation: {}, preview: null,
        resolutions: {}, start: "", stop: "", activeOperation: null};
      reviewSessions.set(annotations, session);
      annotations.onChange(() => session.render());
      selection.onChange(() => session.render());
    }
    container.style.cssText = "padding:10px;box-sizing:border-box;overflow:auto;height:100%;font:13px sans-serif";
    const state = () => annotations.stateRef.current;
    const payload = () => JSON.stringify({annotations: state().annotations});
    if (local && firstMount) {
      // FigPack tracks the baseline of its own toolbar saves, but extensions
      // have no "mark saved" action. Use this control's acknowledged PUT as
      // the local unload baseline, so a saved draft does not trigger a false
      // unsaved-edits warning. Hosted figures retain the upstream handler.
      window.addEventListener("beforeunload", event => {
        if (!session.loaded) return;
        event.stopImmediatePropagation();
        if (payload() !== session.saved) {
          event.preventDefault();
          event.returnValue = "";
        }
      }, {capture: true});
    }
    const curation = () => JSON.parse(state().annotations["/"]?.sorting_curation || JSON.stringify(seed));
    const selected = () => [...selection.stateRef.current.selectedUnitIds].sort((a, b) => Number(a) - Number(b));
    // Merged table rows represent all original contributors until commit.
    const contributors = () => [...new Set(selected().flatMap(id =>
      curation().mergeGroups?.find(group => group.map(String).includes(String(id))) || [id]
    ))];
    const edit = action => {
      session.preview = null;
      reducer.dispatch({type: "SET_CURATION", curation: curation()});
      reducer.dispatch(action);
      const value = JSON.stringify(reducer.stateRef.current);
      annotations.dispatch({type: "setAllAnnotations", annotations: {
        ...state().annotations,
        "/": {...state().annotations["/"], sorting_curation: value}
      }});
    };
    const element = (tag, text, parent = container) => {
      const node = document.createElement(tag);
      if (text !== undefined) node.textContent = text;
      parent.appendChild(node);
      return node;
    };
    const button = (text, action, disabled) => {
      const node = element("button", text);
      node.style.margin = "4px 8px 4px 0";
      node.disabled = disabled;
      node.onclick = action;
      return node;
    };
    const saveDraft = async () => {
      const snapshot = payload();
      if (snapshot === session.saved) return;
      session.saving = true;
      session.error = "";
      session.render();
      try {
        const response = await fetch(new URL("annotations.json", window.location.href), {
          method: "PUT", headers: {"Content-Type": "application/json"}, body: snapshot
        });
        if (!response.ok) throw new Error(`Draft was not saved (HTTP ${response.status}). Retry before committing.`);
        session.saved = snapshot;
      } finally {
        session.saving = false;
        session.render();
      }
    };
    const reportError = error => {
      session.error = error.message;
      session.render();
    };
    const pollOperation = async () => {
      try {
        const response = await fetch("api/operation", {cache: "no-store"});
        if (!response.ok) throw new Error("The Python review session is unavailable. Reopen review.open() to reconnect.");
        const operation = await response.json();
        const changed = JSON.stringify(operation) !== JSON.stringify(session.operation);
        session.operation = operation;
        if (operation.status === "complete" && operation.action === "preview" && changed) {
          session.preview = operation.preview;
          session.resolutions = {};
        }
        if (changed) session.render();
        if (operation.status === "running") {
          setTimeout(pollOperation, 600);
        } else if (operation.status === "complete" && operation.url &&
          session.activeOperation === operation.operation_id &&
          ["commit", "parent"].includes(operation.action)) {
          session.activeOperation = null;
          window.location.assign(operation.url);
        }
      } catch (error) { reportError(error); }
    };
    const operate = async request => {
      try {
        session.error = "";
        if (["preview", "commit", "parent"].includes(request.action)) await saveDraft();
        const response = await fetch("api/operation", {
          method: "POST", headers: {"Content-Type": "application/json", "X-Spyglass-Review": session.reviewId},
          body: JSON.stringify(request)
        });
        const result = await response.json();
        if (!response.ok) throw new Error(result.error || "Review action failed.");
        session.preview = null;
        session.activeOperation = result.operation_id;
        session.operation = {...result, action: request.action, message: "Preparing review operation…"};
        session.render();
        await pollOperation();
      } catch (error) { reportError(error); }
    };
    const render = () => {
      container.replaceChildren();
      const busy = session.operation.status === "running";
      const editable = session.loaded && !busy && (local || state().editingAnnotations);
      const ids = contributors();
      element("div", `Units: ${selected().join(", ") || "none selected"}`);
      const choices = element("div");
      for (const label of palette) {
        const wrapper = element("label", undefined, choices);
        wrapper.style.cssText = "display:inline-block;margin:8px 16px 8px 0";
        const checkbox = element("input", undefined, wrapper);
        checkbox.type = "checkbox";
        checkbox.disabled = !editable || !ids.length;
        const count = ids.filter(id => (curation().labelsByUnit?.[id] || []).includes(label)).length;
        checkbox.checked = !!ids.length && count === ids.length;
        checkbox.indeterminate = count > 0 && count < ids.length;
        checkbox.onchange = () => edit({type: "TOGGLE_UNIT_LABEL", unitId: ids, label});
        element("span", label, wrapper);
      }
      button("Merge Selected", () => edit({type: "MERGE_UNITS", unitIds: ids}), !editable || ids.length < 2);
      button("Undo selected merge", () => edit({type: "UNMERGE_UNITS", unitIds: ids}),
        !editable || !curation().mergeGroups?.some(group => group.some(id => ids.includes(id))));
      if (local) {
        button(session.saving ? "Saving draft…" : "Save draft", () => saveDraft().catch(reportError),
          !session.loaded || session.saving || busy || payload() === session.saved);
      }
      if (session.connected) {
        button("Preview and commit", () => operate({action: "preview"}), !editable || session.saving);
        button("Review parent branch", () => operate({action: "parent"}), !editable || session.saving);
        const inspection = element("div");
        element("span", "Inspection window (recording-relative seconds; blank = full recording): ", inspection);
        for (const [key, name] of [["start", "Start seconds"], ["stop", "Stop seconds"]]) {
          const input = element("input", undefined, inspection);
          input.type = "number";
          input.step = "any";
          input.style.width = "90px";
          input.setAttribute("aria-label", name);
          input.value = session[key];
          input.oninput = () => { session[key] = input.value; };
        }
        button("Inspect selected units / pairs", () => {
          if ((session.start === "") !== (session.stop === "")) {
            reportError(new Error("Enter both start and stop seconds, or leave both blank."));
            return;
          }
          const timeRange = session.start === "" ? null : [Number(session.start), Number(session.stop)];
          if (timeRange && (!timeRange.every(Number.isFinite) || timeRange[0] < 0 || timeRange[1] <= timeRange[0])) {
            reportError(new Error("Choose finite seconds with 0 ≤ start < stop."));
            return;
          }
          operate({action: "inspect", unit_ids: ids,
            time_range: timeRange});
        }, busy || !ids.length);
        element("div", "Detailed inspection includes every selected CCG pair and every raster spike in the window. Trace windows: at most 10 seconds.");
        if (session.preview) {
          const preview = session.preview;
          element("pre", preview.summary).style.whiteSpace = "pre-wrap";
          for (const row of preview.changed_units) {
            element("div", `Unit ${row.unit_id}: labels [${row.labels_before || "none"}] → [${row.labels_after || "none"}]${row.merge_group ? `; merge ${row.merge_group}` : ""}`);
          }
          for (const conflict of preview.conflicts) {
            const resolution = session.resolutions[conflict.merged_unit_id] ||= {labels: [], confirmed: false};
            element("div", `Merged unit ${conflict.merged_unit_id} from ${Object.entries(conflict.contributors).map(([id, labels]) => `${id} [${labels.join(", ") || "none"}]`).join("; ")}`);
            const group = element("div");
            for (const label of conflict.choices) {
              const wrapper = element("label", undefined, group);
              wrapper.style.marginRight = "12px";
              const box = element("input", undefined, wrapper);
              box.type = "checkbox";
              box.checked = resolution.labels.includes(label);
              box.disabled = busy;
              box.onchange = () => {
                resolution.labels = box.checked ? [...resolution.labels, label] : resolution.labels.filter(value => value !== label);
                resolution.confirmed = false;
                session.render();
              };
              element("span", label, wrapper);
            }
            const wrapper = element("label");
            const confirmation = element("input", undefined, wrapper);
            confirmation.type = "checkbox";
            confirmation.checked = resolution.confirmed;
            confirmation.disabled = busy;
            confirmation.onchange = () => { resolution.confirmed = confirmation.checked; session.render(); };
            element("span", "Use these final labels (empty is allowed)", wrapper);
          }
          button(preview.has_merges ? "Commit and inspect merged units" : preview.has_changes ? "Commit curation" : "Record reviewed — no changes",
            () => operate({action: "commit", annotations_hash: preview.annotations_hash,
              confirm_no_changes: !preview.has_changes,
              conflict_resolutions: Object.fromEntries(Object.entries(session.resolutions).map(([id, value]) => [id, value.labels]))}),
            busy || session.saving || preview.conflicts.some(value => !session.resolutions[value.merged_unit_id]?.confirmed));
        }
        if (session.operation.message) element("div", session.operation.message).setAttribute("role", "status");
        if (session.operation.status === "complete" && session.operation.url) {
          const link = element("a", session.operation.action === "inspect" ? "Open selected-unit inspection" : "Continue review");
          link.href = session.operation.url;
          if (session.operation.action === "inspect") { link.target = "_blank"; link.rel = "noopener"; }
        }
        if (session.operation.result && !session.operation.result.needs_merge_verification) {
          element("div", "Ready for analysis. In Python: final_curation = review.result()");
        }
      }
      const status = element("div", session.error || (!session.loaded ? "Loading saved draft…" : local ?
        (payload() === session.saved ? (session.connected ? "Draft saved. Preview and commit when ready." : "Draft saved. Commit in the notebook to create a Spyglass curation.") :
          "Unsaved draft — Save draft or Preview and commit.") :
        "Use Curate Figure, then Save Annotations in the toolbar to save a draft. Commit in the notebook."));
      status.setAttribute("role", "status");
      if (curation().mergeGroups?.length) element("div",
        "Pending merges: displayed waveforms and metrics still describe the original units. Commit and inspect the reevaluated result.");
      if (!session.connected) element("div", "Next: review.commit_panel() previews saved edits and records your review.");
    };
    if (firstMount) {
      const focus = new URLSearchParams(window.location.search).get("spyglass_units");
      let focused = false;
      const focusUnits = () => {
        if (!focused && focus && selection.stateRef.current.orderedUnitIds.length) {
          focused = true;
          const wanted = new Set(focus.split(","));
          selection.dispatch({type: "SET_SELECTION", incomingSelectedUnitIds:
            selection.stateRef.current.orderedUnitIds.filter(id => wanted.has(String(id)))});
        }
      };
      selection.onChange(focusUnits);
      focusUnits();
    }
    session.render = render;
    if (!local) annotations.dispatch({type: "reportViewWithAnnotations"});
    session.render();
    if (local && firstMount) {
      try {
        const response = await fetch(new URL("annotations.json", window.location.href), {cache: "no-store"});
        if (!response.ok && response.status !== 404) throw new Error(`HTTP ${response.status}`);
        const data = response.status === 404 ? {annotations: {"/": {sorting_curation: JSON.stringify(seed)}}} : await response.json();
        annotations.dispatch({type: "setAllAnnotations", annotations: data.annotations || {}});
        session.saved = payload();
        session.loaded = true;
      } catch (exception) {
        session.error = `Unable to load saved annotations: ${exception.message}. Reload before editing.`;
      }
      session.render();
      try {
        const response = await fetch("api/capabilities", {cache: "no-store"});
        if (response.ok) {
          const capabilities = await response.json();
          session.connected = capabilities.connected;
          session.reviewId = capabilities.review_id;
          session.render();
          if (session.connected) await pollOperation();
        }
      } catch (error) { reportError(error); }
    }
  }
});
window.figpack_p1.registerFPExtension({name: "spyglass-review"});
