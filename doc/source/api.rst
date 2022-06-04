=============
API Reference
=============

:mod:`distance3d`
==========================

.. automodule:: distance3d
    :no-members:
    :no-inherited-members:

:mod:`distance3d.distance`
--------------------------

.. automodule:: distance3d.distance
    :no-members:
    :no-inherited-members:

.. currentmodule:: distance3d.distance

.. autosummary::
   :nosignatures:
   :toctree: _autosummary/

   ~distance3d.distance.point_to_line
   ~distance3d.distance.point_to_line_segment
   ~distance3d.distance.point_to_plane
   ~distance3d.distance.point_to_triangle
   ~distance3d.distance.point_to_rectangle
   ~distance3d.distance.point_to_circle
   ~distance3d.distance.point_to_disk
   ~distance3d.distance.point_to_box
   ~distance3d.distance.point_to_cylinder
   ~distance3d.distance.point_to_ellipsoid
   ~distance3d.distance.line_to_line
   ~distance3d.distance.line_to_line_segment
   ~distance3d.distance.line_to_plane
   ~distance3d.distance.line_to_triangle
   ~distance3d.distance.line_to_rectangle
   ~distance3d.distance.line_to_circle
   ~distance3d.distance.line_to_box
   ~distance3d.distance.line_segment_to_line_segment
   ~distance3d.distance.line_segment_to_plane
   ~distance3d.distance.line_segment_to_triangle
   ~distance3d.distance.line_segment_to_rectangle
   ~distance3d.distance.line_segment_to_circle
   ~distance3d.distance.line_segment_to_box
   ~distance3d.distance.plane_to_plane
   ~distance3d.distance.plane_to_triangle
   ~distance3d.distance.plane_to_rectangle
   ~distance3d.distance.plane_to_box
   ~distance3d.distance.plane_to_ellipsoid
   ~distance3d.distance.triangle_to_triangle
   ~distance3d.distance.triangle_to_rectangle
   ~distance3d.distance.rectangle_to_rectangle
   ~distance3d.distance.rectangle_to_box
   ~distance3d.distance.disk_to_disk


:mod:`distance3d.colliders`
---------------------------

.. automodule:: distance3d.colliders
    :no-members:
    :no-inherited-members:

.. currentmodule:: distance3d.colliders

.. autosummary::
   :nosignatures:
   :toctree: _autosummary/

   ~distance3d.colliders.BoundingVolumeHierarchy
   ~distance3d.colliders.ConvexCollider
   ~distance3d.colliders.Convex
   ~distance3d.colliders.MeshGraph
   ~distance3d.colliders.Box
   ~distance3d.colliders.Cylinder
   ~distance3d.colliders.Sphere
   ~distance3d.colliders.Capsule
   ~distance3d.colliders.Ellipsoid


:mod:`distance3d.containment`
-----------------------------

.. automodule:: distance3d.containment
    :no-members:
    :no-inherited-members:

.. currentmodule:: distance3d.containment

.. autosummary::
   :nosignatures:
   :toctree: _autosummary/

   ~distance3d.containment.axis_aligned_bounding_box
   ~distance3d.containment.sphere_aabb
   ~distance3d.containment.box_aabb
   ~distance3d.containment.cylinder_aabb
   ~distance3d.containment.capsule_aabb
   ~distance3d.containment.ellipsoid_aabb


:mod:`distance3d.epa`
---------------------

.. automodule:: distance3d.epa
    :no-members:
    :no-inherited-members:

.. currentmodule:: distance3d.epa

.. autosummary::
   :nosignatures:
   :toctree: _autosummary/

   ~distance3d.epa.epa_vertices
   ~distance3d.epa.epa


:mod:`distance3d.gjk`
---------------------

.. automodule:: distance3d.gjk
    :no-members:
    :no-inherited-members:

.. currentmodule:: distance3d.gjk

.. autosummary::
   :nosignatures:
   :toctree: _autosummary/

   ~distance3d.gjk.gjk
   ~distance3d.gjk.gjk_with_simplex


:mod:`distance3d.mpr`
---------------------

.. automodule:: distance3d.mpr
    :no-members:
    :no-inherited-members:

.. currentmodule:: distance3d.mpr

.. autosummary::
   :nosignatures:
   :toctree: _autosummary/

   ~distance3d.mpr.mpr_intersection
   ~distance3d.mpr.mpr_penetration


:mod:`distance3d.geometry`
--------------------------

.. automodule:: distance3d.geometry
    :no-members:
    :no-inherited-members:

.. currentmodule:: distance3d.geometry

.. autosummary::
   :nosignatures:
   :toctree: _autosummary/

   ~distance3d.geometry.convert_rectangle_to_segment
   ~distance3d.geometry.convert_rectangle_to_vertices
   ~distance3d.geometry.convert_box_to_face
   ~distance3d.geometry.convert_segment_to_line
   ~distance3d.geometry.convert_box_to_vertices
   ~distance3d.geometry.support_function_cylinder
   ~distance3d.geometry.support_function_capsule
   ~distance3d.geometry.support_function_ellipsoid
   ~distance3d.geometry.support_function_box
   ~distance3d.geometry.support_function_sphere
   ~distance3d.geometry.hesse_normal_form
   ~distance3d.geometry.line_from_pluecker


:mod:`distance3d.self_collision`
--------------------------------

.. automodule:: distance3d.self_collision
    :no-members:
    :no-inherited-members:

.. currentmodule:: distance3d.self_collision

.. autosummary::
   :nosignatures:
   :toctree: _autosummary/

   ~distance3d.self_collision.detect
   ~distance3d.self_collision.detect_any


:mod:`distance3d.plotting`
--------------------------

.. automodule:: distance3d.plotting
    :no-members:
    :no-inherited-members:

.. currentmodule:: distance3d.plotting

.. autosummary::
   :nosignatures:
   :toctree: _autosummary/

   ~distance3d.plotting.plot_line
   ~distance3d.plotting.plot_segment
   ~distance3d.plotting.plot_plane
   ~distance3d.plotting.plot_triangle
   ~distance3d.plotting.plot_rectangle
   ~distance3d.plotting.plot_circle
   ~distance3d.plotting.plot_aabb
   ~distance3d.plotting.plot_convex
   ~distance3d.plotting.plot_tetrahedron
   ~distance3d.plotting.plot_aabb_tree


:mod:`distance3d.random`
------------------------

.. automodule:: distance3d.random
    :no-members:
    :no-inherited-members:

.. currentmodule:: distance3d.random

.. autosummary::
   :nosignatures:
   :toctree: _autosummary/

   ~distance3d.random.randn_point
   ~distance3d.random.randn_direction
   ~distance3d.random.randn_line
   ~distance3d.random.randn_line_segment
   ~distance3d.random.randn_plane
   ~distance3d.random.randn_rectangle
   ~distance3d.random.randn_triangle
   ~distance3d.random.rand_box
   ~distance3d.random.rand_capsule
   ~distance3d.random.rand_ellipsoid
   ~distance3d.random.rand_cylinder
   ~distance3d.random.rand_sphere
   ~distance3d.random.randn_convex


:mod:`distance3d.utils`
-----------------------

.. automodule:: distance3d.utils
    :no-members:
    :no-inherited-members:

.. currentmodule:: distance3d.utils

.. autosummary::
   :nosignatures:
   :toctree: _autosummary/

   ~distance3d.utils.norm_vector
