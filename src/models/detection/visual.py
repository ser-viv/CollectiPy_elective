import logging
import math
import numpy as np
from collections import deque 
from core.configuration.plugin_base import DetectionModel
from core.configuration.plugin_registry import register_detection_model
from models.utility_functions import normalize_angle

logger = logging.getLogger("sim.detection.visual")


class VisualDetectionModel(DetectionModel):
    """
    Modello visivo compatibile con il ring attractor.

    Restituisce:
        agents:         campo binario agenti (invariato)
        objects:        campo binario oggetti — ora include i bordi dell'arena circolare
        combined:       max(agents, objects)
        edge_counts:    array (channel_size,) int32 — per ogni spin, quanti bordi
                        angolari cadono nel suo settore. Bordo sx e dx contano separatamente.
        agent_metadata: lista di dict con angle, distance, angular_width,
                        edge_left_visible, edge_right_visible, own_groups,
                        survived_groups per ogni agente percepito (visibile
                        almeno in parte dopo l'occlusione a livello di
                        settore)
        arena_metadata: lista di dict con angle, distance, angular_width per ogni
                        segmento di bordo dell'arena percepito
    """

    def __init__(self, agent, context=None):
        self.agent = agent
        context = context or {}

        self.num_groups = context.get("num_groups", 1)
        self.num_spins_per_group = context.get("num_spins_per_group", 1)
        self.perception_width = context.get("perception_width", 0.5)

        self.group_angles = context.get(
            "group_angles",
            np.linspace(0, 2 * math.pi, self.num_groups, endpoint=False)
        )

        self.reference = context.get("reference", "egocentric")
        self.perception_global_inhibition = context.get("perception_global_inhibition", 0)

        self.max_detection_distance = float(
            context.get("max_detection_distance",
                getattr(self.agent, "perception_distance", math.inf))
        )

        # Numero di punti campionati sul bordo circolare dell'arena.
        # Più alto = rappresentazione più fine ma più costosa.
        # Configurabile via context: "num_boundary_samples"
        self.num_boundary_samples = int(context.get("num_boundary_samples", 36))

        # --- canale di allineamento ---
        self.align_enabled = bool(context.get("align_enabled", False))
        self.align_kappa   = float(context.get("align_kappa", 4.0))
        self.align_source  = str(context.get("align_source", "flow")).lower()   # "flow" | "heading"
        self.flow_lag      = max(1, int(context.get("flow_lag", 10)))           # in campioni di sense()
        self._spin_angles  = np.repeat(np.asarray(self.group_angles, dtype=float),
                                    self.num_spins_per_group)
        self._obs_hist     = {}   # nome vicino -> deque[(phi, w, own_x, own_y_invertita)]

        min_width = 2 * math.pi / self.num_groups
        if self.perception_width < min_width:
            logger.warning(
                "perception_width (%.3f) < angular spacing (%.3f): "
                "blind spots exist between groups",
                self.perception_width, min_width
            )

    def sense(self, agent, objects: dict, agents: dict, arena_shape=None):
        channel_size = self.num_groups * self.num_spins_per_group

        agent_channel       = np.zeros(channel_size)
        object_channel      = np.zeros(channel_size)
        edge_counts         = np.zeros(channel_size, dtype=np.int32)
        # NUOVI canali: già occlusion-aware (z-buffer), popolati agente per
        # agente in _collect_agent_targets. Sostituiscono la vecchia
        # ricostruzione "a blob" fatta a valle in spin_model.py: attrazione
        # (edge) e repulsione (body) vengono ora separate qui, durante la
        # proiezione ottica, prima che l'informazione sui singoli agenti
        # venga persa nell'unione binaria del canale.
        agent_edge_channel = np.zeros(channel_size)
        agent_body_channel = np.zeros(channel_size)
        agent_heading_channel = np.zeros(channel_size)
        agent_metadata = []
        arena_metadata = []

        hierarchy = self._resolve_hierarchy(agent, arena_shape)

        self._collect_agent_targets(
            agent_channel, agents, hierarchy,
            edge_counts, agent_metadata,
            edge_channel=agent_edge_channel,
            body_channel=agent_body_channel,
            heading_channel=agent_heading_channel,
        )
        self._collect_object_targets(object_channel, objects)

        # Bordi dell'arena: popolano object_channel e arena_metadata
        if arena_shape is not None:
            self._collect_arena_boundary(object_channel, arena_shape, arena_metadata)

        self._apply_global_inhibition(agent_channel)
        self._apply_global_inhibition(object_channel)

        combined = np.maximum(agent_channel, object_channel)
        combined = np.clip(combined, 0.0, 1.0)

        return {
            "objects":            object_channel,
            "agents":             agent_channel,
            "combined":           combined,
            "edge_counts":        edge_counts,
            # Canali pronti per essere sommati direttamente a external_field
            # in spin_model.py (vedi update_edge_field / update_body_repulsion_field).
            "agent_edge_channel": agent_edge_channel,
            "agent_body_channel": agent_body_channel,
            "agent_metadata":     agent_metadata,
            "arena_metadata":     arena_metadata,
            "agent_heading_channel": agent_heading_channel,
        }

    def _collect_arena_boundary(self, object_channel, arena_shape, arena_metadata):
        """
        Campiona num_boundary_samples punti equidistanti sul bordo dell'arena
        circolare. Per ciascun punto visibile (distanza <= max_detection_distance)
        accumula il campo binario in object_channel e aggiunge un entry in
        arena_metadata con lo stesso formato di agent_metadata (angle, distance,
        angular_width), compatibile con update_arena_repulsion_field in spin_model.py.

        Il raggio fittizio del punto-bordo è calcolato come la metà dell'arco
        tra due campioni adiacenti, così la copertura angolare è uniforme e
        indipendente dalla distanza.
        """
        TWO_PI = 2.0 * math.pi

        radius = getattr(arena_shape, "radius", None)
        if radius is None:
            # Fallback: prova a ricavarlo dal diametro
            diameter = getattr(arena_shape, "diameter", None)
            if diameter is not None:
                radius = diameter / 2.0
            else:
                logger.warning("arena_shape has no radius or diameter attribute; "
                               "arena boundary perception skipped.")
                return

        # Centro dell'arena — convenzione (0, 0, 0)
        cx = getattr(arena_shape, "center_x", 0.0)
        cy = getattr(arena_shape, "center_y", 0.0)

        # Arco tra campioni adiacenti: usato come "raggio fittizio" del punto
        # per calcolare l'ampiezza angolare percepita (half_subt).
        arc_half = (TWO_PI * radius / self.num_boundary_samples) / 2.0

        agent_x = self.agent.position.x
        agent_y = self.agent.position.y

        for k in range(self.num_boundary_samples):
            theta = TWO_PI * k / self.num_boundary_samples
            bx = cx + radius * math.cos(theta)
            by = cy + radius * math.sin(theta)

            dx = bx - agent_x
            dy = by - agent_y
            distance = math.sqrt(dx * dx + dy * dy)

            if distance > self.max_detection_distance:
                continue

            # Angolo verso il punto-bordo
            angle_world = math.degrees(math.atan2(-dy, dx))

            if self.reference == "egocentric":
                angle = normalize_angle(angle_world - self.agent.orientation.z)
            else:
                angle = normalize_angle(angle_world)

            angle_rad = math.radians(angle)
            if angle_rad < 0:
                angle_rad += TWO_PI

            # Ampiezza angolare: tratta arc_half come raggio apparente del segmento
            half_subt = math.atan(arc_half / max(distance, 1e-6))
            obj_min   = angle_rad - half_subt
            obj_max   = angle_rad + half_subt

            self._accumulate_occlusion_interval(
                object_channel, obj_min, obj_max, strength=1.0
            )

            arena_metadata.append({
                "angle":         angle_rad,
                "distance":      distance,
                "angular_width": 2 * half_subt,
            })

    def _collect_agent_targets(self, perception, agents, hierarchy,
                            edge_counts, agent_metadata,
                            edge_channel=None, body_channel=None,
                            heading_channel=None):
        """
        Raccoglie i bersagli-agente e ne calcola la proiezione angolare con
        occlusione risolta a livello di settore discreto (lo stesso settore
        che il modello percepisce, non un taglio geometrico continuo).

        Il modello non dispone di una vera depth-map continua: l'unica cosa
        che può fare, e che rispecchia la realtà fisica, è questa — se due o
        più agenti sono "allineati", cioè le loro sagome angolari toccano lo
        STESSO settore, in quel settore si vede solo l'agente più vicino
        (la distanza viene calcolata dalle posizioni, che sono comunque note,
        e usata SOLO per dirimere questo conflitto fra settori condivisi, non
        per un ordinamento/taglio geometrico continuo su tutta la scena).

        La transizione fra attrazione (bordo) e repulsione (corpo) dipende
        ORA solo dal numero di settori L occupati dall'impronta NON occlusa
        dell'agente (`_own_footprint_groups`), secondo tre casi geometrici
        (nessuna dipendenza esplicita dalla distanza al di fuori del suo
        effetto su L, e nessuna scala sub-lineare né vincolo di corpo minimo
        forzato):

          - Caso 1 (agente molto lontano): L == 1. L'arco percepito rientra
            in un unico settore, che è interamente un settore di attrazione.
            Nessun corpo/repulsione.
          - Caso 2 (agente lontano, non quanto il Caso 1): L == 2. L'arco è
            suddiviso in due settori, ENTRAMBI di attrazione. Nessun corpo.
          - Caso 3 (agente vicino/molto vicino): L >= 3. L'arco è suddiviso
            in L settori (3 se non troppo vicino, fino a 5-7 se molto
            vicino). I DUE settori estremi dell'arco sono di attrazione,
            tutti i restanti (L - 2) settori centrali sono di repulsione.

        Lo spessore di bordo è quindi sempre fisso a 1 settore per lato
        (vedi `_edge_thickness`): per L <= 2 questo produce naturalmente
        "tutta l'impronta è bordo" (Caso 1 e 2), per L >= 3 produce "solo i
        due estremi sono bordo" (Caso 3).

        Un settore conta come bordo attrattivo di un agente SOLO se rientra
        nello spessore di bordo così calcolato E se sopravvive all'occlusione
        (nessun agente più vicino l'ha già occupato). Un settore "perso" per
        occlusione non genera mai un bordo attrattivo artificiale al suo
        posto: se il settore sopravvive ma non è nello spessore di bordo,
        conta semplicemente come corpo/repulsione. La classificazione
        edge/body di ciascun settore è decisa sull'impronta ORIGINALE
        (pre-occlusione) e poi solo filtrata sui settori sopravvissuti: un
        agente il cui unico settore body sopravvive ma i cui edge sono
        occlusi contribuirà quindi solo repulsione, e viceversa.
        """
        TWO_PI = 2.0 * math.pi
        targets = []

        for club, agent_shapes in agents.items():
            for n, shape in enumerate(agent_shapes):
                meta = getattr(shape, "metadata", {}) if hasattr(shape, "metadata") else {}
                target_name = meta.get("entity_name")

                if target_name:
                    if target_name == self.agent.get_name():
                        continue
                elif f"{club}_{n}" == self.agent.get_name():
                    continue

                target_node = meta.get("hierarchy_node")
                if not self._hierarchy_allows_agent(target_node, hierarchy):
                    continue

                agent_pos = shape.center_of_mass()
                dx = agent_pos.x - self.agent.position.x
                dy = agent_pos.y - self.agent.position.y
                dz = agent_pos.z - self.agent.position.z

                radius   = getattr(shape, "bounding_radius", 0.05)
                distance = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

                if distance > self.max_detection_distance:
                    continue

                angle_world = math.degrees(math.atan2(-dy, dx))

                if self.reference == "egocentric":
                    angle = normalize_angle(angle_world - self.agent.orientation.z)
                else:
                    angle = normalize_angle(angle_world)

                angle_rad = math.radians(angle)
                if angle_rad < 0:
                    angle_rad += TWO_PI

                half_subt = math.atan(radius / max(distance, 1e-6))

                name = target_name or f"{club}_{n}"
                heading = self._estimate_heading(name, meta, angle_rad, 2 * half_subt)
                targets.append({
                    "name":          target_name or f"{club}_{n}",
                    "angle":         angle_rad,
                    "distance":      distance,
                    "obj_min":       angle_rad - half_subt,
                    "obj_max":       angle_rad + half_subt,
                    "angular_width": 2 * half_subt,
                    "heading": heading,
                })

        # --- Occlusione a livello di settore: dal più vicino al più lontano.
        # `claimed` è l'insieme dei settori già "occupati" da un agente più
        # vicino: un agente allineato con un altro sullo stesso settore perde
        # quel settore se non è il più vicino dei due.
        targets.sort(key=lambda t: t["distance"])
        claimed = set()

        n_visible = 0

        for t in targets:
            own_groups = self._own_footprint_groups(t["obj_min"], t["obj_max"])
            if not own_groups:
                continue

            survived = [g for g in own_groups if g not in claimed]

            # Questo bersaglio occlude comunque i successivi (più lontani)
            # con l'intera sua sagoma, visibile o no.
            claimed.update(own_groups)

            if not survived:
                # Completamente allineato dietro un agente più vicino su
                # tutti i suoi settori: invisibile, nessun contributo.
                continue

            n_visible += 1
            if heading_channel is not None and t["heading"] is not None:
                self._add_heading_bump(heading_channel, t["heading"])

            # Spessore di bordo (in settori, per lato): SEMPRE 1, indipendente
            # da L e dalla distanza (vedi `_edge_thickness`). Per L <= 2
            # questo copre l'intera impronta (Caso 1 e 2, tutta attrazione);
            # per L >= 3 copre solo i due settori estremi (Caso 3).
            L = len(own_groups)
            w = self._edge_thickness(L)
            true_edge_groups = set(own_groups[:w]) | set(own_groups[-w:])

            survived_set          = set(survived)
            edge_groups_survived  = true_edge_groups & survived_set
            body_groups_survived  = survived_set - true_edge_groups

            self._mark_groups(perception, survived, strength=1.0)

            if body_channel is not None and body_groups_survived:
                self._mark_groups(body_channel, body_groups_survived, strength=1.0)
            if edge_channel is not None and edge_groups_survived:
                self._mark_groups(edge_channel, edge_groups_survived, strength=1.0)
                self._increment_groups(edge_counts, edge_groups_survived)

            left_edge_survived  = set(own_groups[:w]) & survived_set
            right_edge_survived = set(own_groups[-w:]) & survived_set

            # metadati
            agent_metadata.append({
                "name":                 t["name"],
                "angle":                t["angle"],
                "distance":             t["distance"],
                "angular_width":        t["angular_width"],
                "footprint_sectors":    L,
                "edge_thickness":       w,
                "edge_left_visible":    bool(left_edge_survived),
                "edge_right_visible":   bool(right_edge_survived),
                "own_groups":           own_groups,
                "survived_groups":      survived,
                "edge_groups_survived": sorted(edge_groups_survived),
                "body_groups_survived": sorted(body_groups_survived),
                "heading_est": t["heading"],
            })

        if heading_channel is not None and n_visible > 0:
            heading_channel /= n_visible

    def _edge_thickness(self, L):
        """
        Spessore di bordo (in settori, per lato): SEMPRE fisso a 1 settore,
        indipendentemente da L (larghezza angolare totale della sagoma
        dell'agente osservato, in settori) e quindi dalla sua distanza.

        Con w=1, combinato con `own_groups[:w]` e `own_groups[-w:]` in
        `_collect_agent_targets`:
          - L == 1 (Caso 1, agente molto lontano): il primo e l'ultimo
            elemento coincidono con l'unico settore dell'impronta, quindi
            l'intero (unico) settore è attrazione. Nessun corpo.
          - L == 2 (Caso 2, agente lontano): i due settori dell'impronta
            sono, rispettivamente, il "primo" e l'"ultimo" elemento, quindi
            ENTRAMBI risultano attrazione. Nessun corpo.
          - L >= 3 (Caso 3, agente vicino/molto vicino): solo i due settori
            estremi dell'impronta sono attrazione; i restanti (L - 2)
            settori centrali sono repulsione (corpo). Più l'agente è
            vicino, più L cresce, e quindi cresce il numero di settori di
            repulsione centrali.

        Nessuna scala sub-lineare (era sqrt(L)) e nessun clamp che
        garantisse un corpo minimo forzato: la vecchia estensione forzata
        dell'impronta a un minimo di 3 settori (`_extend_footprint`) è
        stata rimossa da `_own_footprint_groups` proprio perché generava un
        body artificiale anche a distanze in cui l'agente osservato
        dovrebbe produrre solo attrazione (Caso 1 e 2), impedendo la
        polarizzazione nel flocking.
        """
        return 1
    
    def reset_flow(self):
        """Azzera la memoria del flusso ottico (nuovo run)."""
        self._obs_hist = {}

    def _add_heading_bump(self, channel, heading):
        """Bump von Mises centrato su `heading` (rad, frame del ring)."""
        channel += np.exp(self.align_kappa * (np.cos(self._spin_angles - heading) - 1.0))

    def _estimate_heading(self, name, meta, phi, w):
        """Heading del vicino nel frame del ring, o None se non disponibile."""
        if not self.align_enabled:
            return None
        if self.align_source == "heading":
            oz = meta.get("orientation_z")
            return None if oz is None else math.radians(oz) % (2.0 * math.pi)
        p = self.agent.position
        h = self._obs_hist.get(name)
        if h is None:
            h = self._obs_hist[name] = deque(maxlen=self.flow_lag + 1)
        # frame del ring: (x, -y), coerente con angle_world = atan2(-dy, dx)
        h.append((phi, w, p.x, -p.y))
        return self._flow_heading(h)

    def _flow_heading(self, hist):
        """Stima dell'heading dal moto visivo (bearing + ampiezza angolare), senza distanza."""
        if len(hist) <= self.flow_lag:
            return None
        phi0, w0, x0, y0 = hist[0]
        phi1, w1, x1, y1 = hist[-1]
        dphi = (phi1 - phi0 + math.pi) % (2 * math.pi) - math.pi
        pm   = phi0 + dphi / 2
        dw   = (w1 - w0) / (0.5 * (w0 + w1))
        # rho = dphi * e_perp - (dw) * e_r  (direzione di v_j - v_i, scalata da 1/d)
        rho = np.array([-math.sin(pm) * dphi - math.cos(pm) * dw,
                        math.cos(pm) * dphi - math.sin(pm) * dw])
        n2 = float(rho @ rho)
        if n2 < 1e-14:                       # vicino già allineato: nessun segnale
            return None
        vx, vy = x1 - x0, y1 - y0            # spostamento proprio sulla stessa finestra
        nv = math.hypot(vx, vy)
        if nv < 1e-12:
            return None
        ui = np.array([vx / nv, vy / nv])
        k = -2.0 * float(ui @ rho) / n2      # da |u_j| = 1
        if k <= 0:
            return None
        uj = ui + k * rho
        return math.atan2(uj[1], uj[0])

    def _increment_groups(self, edge_counts, groups):
        """Incrementa di 1 gli slot degli spin corrispondenti ai gruppi indicati."""
        for g in groups:
            start = g * self.num_spins_per_group
            end   = start + self.num_spins_per_group
            edge_counts[start:end] += 1

    def _accumulate_edge(self, edge_counts, edge_angle):
        """Incrementa di 1 gli spin il cui settore contiene edge_angle."""
        TWO_PI = 2.0 * math.pi
        edge_angle = edge_angle % TWO_PI

        for g, center in enumerate(self.group_angles):
            sec_min   = (center - self.perception_width / 2) % TWO_PI
            sec_max   = (center + self.perception_width / 2) % TWO_PI
            sec_wraps = sec_min > sec_max

            if sec_wraps:
                contains = (edge_angle >= sec_min) or (edge_angle <= sec_max)
            else:
                contains = sec_min <= edge_angle <= sec_max

            if contains:
                start = g * self.num_spins_per_group
                end   = start + self.num_spins_per_group
                edge_counts[start:end] += 1

    def _collect_object_targets(self, perception, objects):
        TWO_PI = 2.0 * math.pi
        for _, (shapes, positions, strengths, uncertainties) in objects.items():
            for i in range(len(shapes)):
                dx = positions[i].x - self.agent.position.x
                dy = positions[i].y - self.agent.position.y
                dz = positions[i].z - self.agent.position.z

                radius   = getattr(shapes[i], "bounding_radius", 0.05)
                distance = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

                if distance > self.max_detection_distance:
                    continue

                angle_world = math.degrees(math.atan2(-dy, dx))

                if self.reference == "egocentric":
                    angle = normalize_angle(angle_world - self.agent.orientation.z)
                else:
                    angle = normalize_angle(angle_world)

                angle_rad = math.radians(angle)
                if angle_rad < 0:
                    angle_rad += TWO_PI

                half_subt = math.atan(radius / max(distance, 1e-6))
                obj_min   = angle_rad - half_subt
                obj_max   = angle_rad + half_subt

                self._accumulate_occlusion_interval(
                    perception, obj_min, obj_max, strength=1.0
                )

    def _apply_global_inhibition(self, perception_channel):
        if self.perception_global_inhibition == 0:
            return
        perception_channel -= self.perception_global_inhibition

    def _accumulate_occlusion_interval(self, perception, obj_min, obj_max, strength=1.0):
        groups = self._groups_intersecting_interval(obj_min, obj_max)
        self._mark_groups(perception, groups, strength=strength)

    def _groups_intersecting_interval(self, obj_min, obj_max):
        """Ritorna gli indici di gruppo (settore angolare) che intersecano
        l'intervallo [obj_min, obj_max] (gestisce il wrap-around a 2*pi)."""
        TWO_PI = 2.0 * math.pi

        obj_min   = obj_min % TWO_PI
        obj_max   = obj_max % TWO_PI
        obj_wraps = obj_min > obj_max

        groups = []
        for g, center in enumerate(self.group_angles):
            sec_min   = (center - self.perception_width / 2) % TWO_PI
            sec_max   = (center + self.perception_width / 2) % TWO_PI
            sec_wraps = sec_min > sec_max

            if not obj_wraps and not sec_wraps:
                intersects = not (sec_max < obj_min or sec_min > obj_max)
            elif obj_wraps and not sec_wraps:
                intersects = (sec_max >= obj_min) or (sec_min <= obj_max)
            elif not obj_wraps and sec_wraps:
                intersects = (obj_max >= sec_min) or (obj_min <= sec_max)
            else:
                intersects = True

            if intersects:
                groups.append(g)

        return groups

    def _sector_indices_for_angle(self, angle):
        """Ritorna gli indici di gruppo il cui settore contiene esattamente
        `angle` (usato per individuare in quale settore cade un bordo)."""
        return self._groups_intersecting_interval(angle, angle)

    def _own_footprint_groups(self, obj_min, obj_max):
        """
        Ritorna la lista dei settori toccati dalla sagoma angolare propria
        di un agente (obj_min, obj_max), PRIMA di qualunque occlusione,
        ordinata circolarmente dal settore di obj_min a quello di obj_max
        (gestisce il wrap-around a 2*pi). Il primo e l'ultimo elemento sono
        i due settori-bordo veri di questo agente (o l'unico settore, se
        L == 1).

        La sagoma di un singolo agente è per costruzione un unico arco
        contiguo (corpo circolare), quindi basta camminare in avanti di
        settore in settore dal bordo sinistro fino a raggiungere il bordo
        destro.

        La lunghezza risultante L = len(ordered) riflette fedelmente
        l'ampiezza angolare reale percepita (nessuna estensione artificiale
        viene più applicata):

          - L == 1: agente molto lontano (Caso 1) — un unico settore,
            interamente di attrazione (vedi `_edge_thickness`).
          - L == 2: agente lontano (Caso 2) — due settori, entrambi di
            attrazione.
          - L >= 3: agente vicino/molto vicino (Caso 3) — i due settori
            estremi sono di attrazione, i restanti (L - 2) di repulsione.

        In precedenza, quando L < 3, l'impronta veniva estesa
        simmetricamente a un minimo di 3 settori per garantire sempre
        almeno 1 settore centrale di corpo. Questa estensione è stata
        rimossa su richiesta esplicita: generava un corpo artificiale anche
        a grande distanza, dove invece l'agente dovrebbe produrre solo
        attrazione, impedendo la polarizzazione nel flocking (alta
        coesione, bassa polarizzazione osservata nelle simulazioni).
        """
        TWO_PI = 2.0 * math.pi
        all_groups = set(self._groups_intersecting_interval(obj_min, obj_max))
        if not all_groups:
            return []

        start_candidates = self._sector_indices_for_angle(obj_min % TWO_PI)
        start = start_candidates[0] if start_candidates else min(all_groups)

        ordered = []
        g = start
        for _ in range(self.num_groups):
            if g in all_groups and g not in ordered:
                ordered.append(g)
            if len(ordered) == len(all_groups):
                break
            g = (g + 1) % self.num_groups

        return ordered

    def _mark_groups(self, channel, groups, strength=1.0):
        """Scrive `strength` nei blocchi di spin corrispondenti ai gruppi
        indicati (stessa granularità usata per il campo binario storico)."""
        for g in groups:
            start = g * self.num_spins_per_group
            end   = start + self.num_spins_per_group
            channel[start:end] = strength

    @staticmethod
    def _interval_intersection(a1, a2, b1, b2):
        left  = max(a1, b1)
        right = min(a2, b2)
        return max(0.0, right - left)

    @staticmethod
    def _resolve_hierarchy(agent, arena_shape):
        if arena_shape is not None:
            metadata = getattr(arena_shape, "metadata", None)
            if metadata:
                hierarchy = metadata.get("hierarchy")
                if hierarchy:
                    return hierarchy
        return getattr(agent, "hierarchy_context", None)

    def _hierarchy_allows_agent(self, target_node, hierarchy) -> bool:
        checker = getattr(self.agent, "allows_hierarchical_link", None)
        if not callable(checker):
            return True
        return bool(checker(target_node, "detection", hierarchy))


register_detection_model("VISUAL", lambda agent, context=None: VisualDetectionModel(agent, context))