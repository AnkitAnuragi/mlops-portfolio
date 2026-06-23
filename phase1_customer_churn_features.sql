-- Customer-level feature table, ready for ML consumption (Phase 2+).
-- One row per customer, combining signup attributes with engineered
-- features derived from their usage events in the trailing 30 days.

with customers as (
    select * from {{ ref('stg_customers') }}
),

events as (
    select * from {{ ref('stg_events') }}
),

login_features as (
    select
        customer_id,
        count(*) as login_count_30d,
        min(days_ago) as days_since_last_login
    from events
    where event_type = 'login'
    group by customer_id
),

ticket_features as (
    select
        customer_id,
        count(*) as support_tickets_30d
    from events
    where event_type = 'support_ticket'
    group by customer_id
)

select
    c.customer_id,
    c.signup_plan,
    c.tenure_days,
    c.monthly_charge,
    c.age,
    coalesce(l.login_count_30d, 0)          as login_count_30d,
    coalesce(l.days_since_last_login, 30)   as days_since_last_login,
    coalesce(t.support_tickets_30d, 0)      as support_tickets_30d,
    -- engagement ratio: tickets per login, a proxy for friction
    case
        when coalesce(l.login_count_30d, 0) = 0 then coalesce(t.support_tickets_30d, 0)
        else round(coalesce(t.support_tickets_30d, 0)::float / l.login_count_30d, 3)
    end as ticket_to_login_ratio,
    c.churned
from customers c
left join login_features l on c.customer_id = l.customer_id
left join ticket_features t on c.customer_id = t.customer_id
