

CREATE TABLE public.address_info (
    address_id integer NOT NULL,
    street_name text,
    zipcode integer,
    latitude double precision,
    longitude double precision,
    neighborhood_name text,
    tax_district_name integer,
    house_number integer
);

CREATE TABLE public.exterior_info (
    exterior_id integer NOT NULL,
    acres double precision,
    wall_description text,
    construction_grade text,
    year_built integer
);

CREATE TABLE public.interior_info (
    interior_id integer NOT NULL,
    num_units integer,
    story_1_area double precision,
    story_2_area double precision,
    heat_type integer,
    basement_type integer,
    num_bed integer,
    num_bath integer,
    num_kitchen integer
);

CREATE TABLE public.owner_info (
    owner_id integer NOT NULL,
    owner_1 text,
    owner_2 text
);

CREATE TABLE public.properties (
    sbl text NOT NULL,
    width integer,
    depth integer,
    address_id integer,
    exterior_id integer,
    interior_id integer,
    owner_id integer,
    land_value numeric,
    total_value numeric,
    sale_price numeric
);
